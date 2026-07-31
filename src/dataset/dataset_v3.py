from transformers import AutoTokenizer
import json
import logging
import math
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch
try:
    from torch_geometric.data import Data as PyGData
except ImportError:
    PyGData = None


RELATION_NONE = 0
RELATION_SAME_SPEAKER = 1
RELATION_CROSS_REPLY = 2
RELATION_WEAK_AGREE = 3
RELATION_STRONG_AGREE = 4
RELATION_WEAK_CHALLENGE = 5
RELATION_STRONG_CHALLENGE = 6
RELATION_QUESTION = 7

RELATION_AGREE_CUE = RELATION_STRONG_AGREE
RELATION_CHALLENGE_CUE = RELATION_STRONG_CHALLENGE
RELATION_QUESTION_CUE = RELATION_QUESTION

EDGE_SELF = 0
EDGE_NEXT_TURN = 1
EDGE_REPLY = 2
EDGE_SAME_SPEAKER = 3
EDGE_SPEAKER_TO_UTT = 4
EDGE_TARGET_TO_UTT = 5
EDGE_AGREE_REPLY = 6
EDGE_CHALLENGE_REPLY = 7
EDGE_QUESTION_REPLY = 8
EDGE_ROOT_TO_UTT = 9

EDGE_GROUP_CONTEXT = 0
EDGE_GROUP_SPEAKER_HISTORY = 1
EDGE_GROUP_AUXILIARY = 2

CONTEXT_EDGE_TYPES = {
    EDGE_NEXT_TURN,
    EDGE_REPLY,
    EDGE_AGREE_REPLY,
    EDGE_CHALLENGE_REPLY,
    EDGE_QUESTION_REPLY,
}

EDGE_TYPE_WEIGHTS = {
    EDGE_NEXT_TURN: 0.5,
    EDGE_REPLY: 1.0,
    EDGE_AGREE_REPLY: 1.2,
    EDGE_QUESTION_REPLY: 0.6,
    EDGE_CHALLENGE_REPLY: 0.3,
    EDGE_SAME_SPEAKER: 1.0,
    EDGE_SELF: 1.0,
}

MODEL_KNOWLEDGE_FIELDS = (
    'description',
    'favor_reason',
    'against_reason',
)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class DataProcessor():
    """
    v3 dataset loader. Target knowledge is read from a local JSON at init time only.
    Training never performs web search or live knowledge retrieval.
    """

    MODEL_KNOWLEDGE_FIELDS = MODEL_KNOWLEDGE_FIELDS

    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_dir)
        self.config = config
        self.use_target_knowledge = bool(getattr(config, 'use_target_knowledge', 1))
        self.target_knowledge = self.load_target_knowledge()
        self.target_knowledge_fields = self.load_target_knowledge_fields()
        self._log_and_validate_target_knowledge()

    @staticmethod
    def _truncate_field(text, max_chars):
        text = str(text or '').strip()
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + '…'

    @classmethod
    def normalize_model_entry(cls, entry):
        if not isinstance(entry, dict):
            return {field: '' for field in MODEL_KNOWLEDGE_FIELDS}
        if all(field in entry for field in MODEL_KNOWLEDGE_FIELDS):
            return {field: str(entry.get(field, '')).strip() for field in MODEL_KNOWLEDGE_FIELDS}

        def join_block(block, key='reasons'):
            if not isinstance(block, dict):
                return ''
            values = block.get(key) or block.get('possible_attitudes') or []
            if isinstance(values, str):
                return values.strip()
            return '；'.join(str(x).strip() for x in values[:2] if str(x).strip())

        favor = entry.get('favor', {})
        against = entry.get('against', {})
        neutral = entry.get('neutral', {})
        description = str(entry.get('description', '')).strip()
        scope = str(entry.get('scope_note', '')).strip()
        if scope:
            description = (description + ' ' + scope).strip()
        return {
            'description': description,
            'favor_reason': join_block(favor),
            'against_reason': join_block(against),
        }

    @classmethod
    def compress_stance_knowledge(cls, fields, max_total=64):
        side_budget = max(max_total // 2, 20)
        segments = []
        used = 0
        for key, label in [
            ('favor_reason', '可能支持'),
            ('against_reason', '可能反对'),
        ]:
            raw = str(fields.get(key, '')).strip()
            if not raw:
                continue
            cap = min(side_budget, max_total - used)
            if cap <= 0:
                break
            clipped = cls._truncate_field(raw, cap)
            segment = f'{label}：{clipped}'
            if used + len(segment) > max_total:
                segment = cls._truncate_field(segment, max_total - used)
            if not segment:
                break
            segments.append(segment)
            used += len(segment)
        return ' '.join(segments).strip()

    @classmethod
    def compress_knowledge_card(cls, fields, max_total=200, stance_only=False):
        if stance_only:
            return cls.compress_stance_knowledge(fields, max_total=max_total)
        desc_budget = min(60, max(40, max_total // 3))
        side_budget = min(45, max(30, (max_total - desc_budget) // 3))
        segments = []
        used = 0
        for key, label in [
            ('description', '说明'),
            ('favor_reason', '可能支持'),
            ('against_reason', '可能反对'),
        ]:
            raw = str(fields.get(key, '')).strip()
            if not raw:
                continue
            cap = desc_budget if key == 'description' else side_budget
            cap = min(cap, max_total - used)
            if cap <= 0:
                break
            clipped = cls._truncate_field(raw, cap)
            segment = f'{label}：{clipped}'
            if used + len(segment) > max_total:
                segment = cls._truncate_field(segment, max_total - used)
            if not segment:
                break
            segments.append(segment)
            used += len(segment)
        return ' '.join(segments).strip()

    def _entry_has_content(self, entry):
        if isinstance(entry, str):
            return bool(entry.strip())
        if isinstance(entry, dict):
            fields = self.normalize_model_entry(entry)
            return any(fields[k] for k in MODEL_KNOWLEDGE_FIELDS)
        return False

    def load_target_knowledge(self):
        if not self.use_target_knowledge:
            logging.info('[Target Knowledge] disabled (use_target_knowledge=0)')
            return {}

        path = getattr(self.config, 'target_knowledge_path', '')
        if not path:
            logging.warning('[Target Knowledge] target_knowledge_path is empty; using target name fallback only')
            return {}
        if not os.path.exists(path):
            logging.warning('[Target Knowledge] file not found: %s; using target name fallback only', path)
            return {}

        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        max_total = int(getattr(self.config, 'target_knowledge_max_chars', 200))
        stance_only = bool(getattr(self.config, 'knowledge_stance_only', 0))
        knowledge = {}
        empty_targets = []
        for target, value in raw.items():
            if not self._entry_has_content(value):
                empty_targets.append(str(target))
                continue
            if isinstance(value, str):
                knowledge[str(target)] = self._truncate_field(value, max_total)
                continue
            fields = self.normalize_model_entry(value)
            knowledge[str(target)] = self.compress_knowledge_card(
                fields, max_total=max_total, stance_only=stance_only,
            )

        logging.info('[Target Knowledge] loaded %d non-empty targets from %s', len(knowledge), path)
        if empty_targets:
            logging.warning(
                '[Target Knowledge] skipped %d empty entries (examples: %s)',
                len(empty_targets),
                empty_targets[:5],
            )
        return knowledge

    def load_target_knowledge_fields(self):
        if not self.use_target_knowledge:
            return {}

        path = getattr(self.config, 'target_knowledge_path', '')
        if not path or not os.path.exists(path):
            return {}

        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        fields_by_target = {}
        max_side = int(getattr(self.config, 'target_knowledge_max_chars', 64))
        for target, value in raw.items():
            if isinstance(value, str):
                fields_by_target[str(target)] = {
                    'description': self._truncate_field(value, max_side),
                    'favor_reason': '',
                    'against_reason': '',
                }
                continue
            if not isinstance(value, dict):
                continue
            fields = self.normalize_model_entry(value)
            fields_by_target[str(target)] = {
                'description': self._truncate_field(fields.get('description', ''), max_side),
                'favor_reason': self._truncate_field(fields.get('favor_reason', ''), max_side),
                'against_reason': self._truncate_field(fields.get('against_reason', ''), max_side),
            }
        return fields_by_target

    def get_knowledge_stance_texts(self, dialogue):
        target = str(dialogue['target'])
        if not self.use_target_knowledge:
            return '', '', ''

        fields = self.target_knowledge_fields.get(target, {})
        favor = fields.get('favor_reason', '')
        against = fields.get('against_reason', '')
        neutral = fields.get('description', '')
        if not any([favor, against, neutral]):
            fallback = self.target_knowledge.get(target, f'target={target}')
            return fallback, fallback, fallback
        return favor, against, neutral

    @classmethod
    def _tokenize_knowledge_side(cls, tokenizer, target, text, side_label, max_len):
        if not str(text).strip():
            return []
        know_text = f'[CLS]关于[SEP]{target}[SEP]{side_label}：{text}[SEP]'
        tokens = tokenizer.tokenize(know_text)
        if max_len > 0 and len(tokens) > max_len:
            tokens = tokens[: max_len - 1] + ['[SEP]']
        return tokens

    def collect_dataset_targets(self):
        targets = set()
        for path_key in ('train_path', 'dev_path', 'test_path'):
            path = getattr(self.config, path_key, '')
            if not path or not os.path.exists(path):
                continue
            with open(path, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    targets.add(str(item['target']))
        return targets

    def _log_and_validate_target_knowledge(self):
        if not self.use_target_knowledge:
            return

        all_targets = self.collect_dataset_targets()
        if not all_targets:
            return

        missing = all_targets - set(self.target_knowledge.keys())
        covered = len(all_targets) - len(missing)
        logging.info(
            '[Target Knowledge] coverage: %d/%d (%.1f%%)',
            covered,
            len(all_targets),
            100.0 * covered / len(all_targets),
        )
        if missing:
            logging.warning(
                '[Target Knowledge] missing for %d targets, examples: %s',
                len(missing),
                sorted(missing)[:10],
            )

        if bool(getattr(self.config, 'require_target_knowledge', 0)) and missing:
            raise ValueError(
                f'Missing target knowledge for {len(missing)} targets; '
                f'examples: {sorted(missing)[:10]}'
            )

    def get_knowledge_text(self, dialogue):
        target = str(dialogue['target'])
        target_type = dialogue.get('target_type', '')
        if not self.use_target_knowledge:
            return f'target={target}; target_type={target_type}'

        text = self.target_knowledge.get(target, '')
        if not text:
            text = f'target={target}; target_type={target_type}'
        max_chars = int(getattr(self.config, 'target_knowledge_max_chars', 200))
        return text[:max_chars] if max_chars > 0 else text

    def get_mask_position(self, tokens):
        mask_token = self.tokenizer.mask_token or '[MASK]'
        if mask_token in tokens:
            return tokens.index(mask_token)
        if '[MASK]' in tokens:
            return tokens.index('[MASK]')
        raise ValueError('Cannot find [MASK] token in tokenized sentence')

    @staticmethod
    def _truncate_tokens_keep_mask(tokens, max_seq_length, mask_token='[MASK]'):
        """Truncate long sequences without dropping the [MASK] prediction token."""
        if max_seq_length <= 0 or len(tokens) <= max_seq_length:
            return tokens

        mask_aliases = {mask_token, '[MASK]'}
        mask_idx = next((i for i, tok in enumerate(tokens) if tok in mask_aliases), None)
        if mask_idx is None:
            return tokens[: max_seq_length - 1] + ['[SEP]']

        suffix = tokens[mask_idx:]
        prefix = tokens[:mask_idx]
        budget = max_seq_length - len(suffix)
        if budget <= 0:
            return suffix[:max_seq_length]
        if len(prefix) <= budget:
            return prefix + suffix

        # Keep [CLS] + utterance head + template tail (target / 立场为) before [MASK].
        keep_head = min(8, budget // 2, len(prefix))
        keep_tail = min(budget - keep_head, 24, len(prefix) - keep_head)
        if keep_tail <= 0:
            return prefix[:budget] + suffix

        head = prefix[:keep_head]
        tail = prefix[-keep_tail:]
        merged = head + tail
        if len(merged) > budget:
            merged = merged[:budget]
        return merged + suffix

    def _is_question(self, text, question_cues):
        text = str(text)
        if '?' in text or '？' in text:
            return True
        return any(cue in text for cue in question_cues if cue not in {'?', '？'})

    def infer_reply_relation(self, current, previous, same_speaker):
        if previous is None:
            return RELATION_NONE
        current = str(current)
        if same_speaker:
            return RELATION_SAME_SPEAKER

        strong_agree_cues = ['确实', '同意', '支持', '赞同', '没错', '说得对', '没毛病', '赞']
        weak_agree_cues = ['是的', '对啊', '对呀', '对的', '有道理']
        strong_challenge_cues = [
            '不是', '不能', '不会', '不对', '不可能', '不行', '错了', '错误',
            '别吹', '别洗', '别尬', '尬黑', '洗地', '喷子', '瞎', '扯'
        ]
        weak_challenge_cues = [
            '不用', '不需要', '不发展', '不懂', '不看好', '没用', '没什么', '没有', '没必要',
            '别说', '别拿', '高估', '笑死', '呵呵', '凭什么', '拉倒',
            '问题是', '但是', '然而', '可是', '反而', '难道', '你懂', '知道什么',
            '有什么关系', '有必要', '又不会死', '怎么就'
        ]
        question_cues = [
            '?', '？', '吗', '么', '呢', '为什么', '怎么', '如何', '谁', '哪',
            '是不是', '有没有', '能不能', '你说得对', '对吗', '对吧'
        ]

        if any(cue in current for cue in strong_challenge_cues):
            return RELATION_STRONG_CHALLENGE
        if any(cue in current for cue in strong_agree_cues):
            return RELATION_STRONG_AGREE
        if self._is_question(current, question_cues):
            return RELATION_QUESTION
        if any(cue in current for cue in weak_challenge_cues):
            return RELATION_WEAK_CHALLENGE
        if any(cue in current for cue in weak_agree_cues):
            return RELATION_WEAK_AGREE
        return RELATION_CROSS_REPLY

    def infer_reply_parent(self, turn_id, speakers, window=4):
        parent, _ = self.infer_reply_parent_with_confidence(turn_id, speakers, window=window)
        return parent

    def infer_reply_parent_with_confidence(self, turn_id, speakers, window=4):
        if turn_id <= 0:
            return -1, 0.45
        current_speaker = speakers[turn_id]
        previous_speaker = speakers[turn_id - 1]
        if previous_speaker != current_speaker:
            return turn_id - 1, 0.90

        start = max(0, turn_id - int(window))
        for idx in range(turn_id - 1, start - 1, -1):
            if speakers[idx] != current_speaker:
                return idx, 0.90
        return turn_id - 1, 0.65

    def reply_relation_to_edge_type(self, relation, same_speaker_parent):
        if same_speaker_parent:
            return EDGE_SAME_SPEAKER
        if relation in (RELATION_STRONG_AGREE, RELATION_WEAK_AGREE):
            return EDGE_AGREE_REPLY
        if relation in (RELATION_STRONG_CHALLENGE, RELATION_WEAK_CHALLENGE):
            return EDGE_CHALLENGE_REPLY
        if relation == RELATION_QUESTION:
            return EDGE_QUESTION_REPLY
        if relation == RELATION_SAME_SPEAKER:
            return EDGE_SAME_SPEAKER
        return EDGE_REPLY

    def reply_chain(self, turn_id, reply_parents, max_depth=6):
        chain = [turn_id]
        seen = {turn_id}
        parent = reply_parents[turn_id] if turn_id < len(reply_parents) else turn_id - 1
        while 0 <= parent < chain[-1] and parent not in seen and len(chain) < max_depth:
            chain.append(parent)
            seen.add(parent)
            parent = reply_parents[parent] if parent < len(reply_parents) else parent - 1
        chain.reverse()
        return chain

    def edge_type_to_group(self, edge_type):
        if edge_type == EDGE_SAME_SPEAKER:
            return EDGE_GROUP_SPEAKER_HISTORY
        if edge_type in CONTEXT_EDGE_TYPES:
            return EDGE_GROUP_CONTEXT
        return EDGE_GROUP_AUXILIARY

    def edge_type_weight(self, edge_type):
        return EDGE_TYPE_WEIGHTS.get(edge_type, 1.0)

    def build_topology_graph(self, speakers, reply_relations, reply_parents, reply_confidences=None, local_window=3):
        """Build utterance graph: context edges + speaker_ids for hypergraph channel."""
        speaker_ids = [int(speaker) for speaker in speakers]
        num_turns = len(speaker_ids)
        edge_time_decay = float(getattr(self.config, 'edge_time_decay', 0.3))

        edge_src = []
        edge_dst = []
        edge_types = []
        edge_groups = []
        edge_weights = []
        added_pairs = set()

        def decay_weight(src, dst, base=1.0):
            distance = max(dst - src, 0)
            return float(base) * math.exp(-edge_time_decay * distance)

        def add_edge(src, dst, edge_type, edge_group, weight=1.0):
            if not (0 <= src < num_turns and 0 <= dst < num_turns):
                return
            weight = float(weight) * self.edge_type_weight(edge_type)
            if src != dst:
                pair = (src, dst)
                if pair in added_pairs:
                    return
                added_pairs.add(pair)
            edge_src.append(src)
            edge_dst.append(dst)
            edge_types.append(edge_type)
            edge_groups.append(edge_group)
            edge_weights.append(weight)

        for turn_id, speaker in enumerate(speaker_ids):
            add_edge(turn_id, turn_id, EDGE_SELF, EDGE_GROUP_AUXILIARY, 1.0)

            if turn_id > 0:
                prev = turn_id - 1
                if speaker_ids[prev] != speaker:
                    add_edge(
                        prev,
                        turn_id,
                        EDGE_NEXT_TURN,
                        EDGE_GROUP_CONTEXT,
                        decay_weight(prev, turn_id),
                    )

            parent = reply_parents[turn_id] if turn_id < len(reply_parents) else -1
            confidence = reply_confidences[turn_id] if reply_confidences is not None else 1.0
            if 0 <= parent < turn_id:
                relation = reply_relations[turn_id] if turn_id < len(reply_relations) else RELATION_CROSS_REPLY
                same_speaker_parent = speaker_ids[parent] == speaker
                edge_type = EDGE_SAME_SPEAKER if same_speaker_parent else self.reply_relation_to_edge_type(relation, False)
                edge_group = EDGE_GROUP_CONTEXT
                add_edge(
                    parent,
                    turn_id,
                    edge_type,
                    edge_group,
                    decay_weight(parent, turn_id, base=confidence),
                )

        assert len(edge_types) == len(edge_groups) == len(edge_weights), 'edge metadata length mismatch'

        graph_values = {
            'num_utterance_nodes': num_turns,
            'num_nodes': num_turns,
            'edge_index': [edge_src, edge_dst],
            'edge_type': edge_types,
            'edge_types': edge_types,
            'edge_group': edge_groups,
            'edge_weight': edge_weights,
            'speaker_ids_for_turn': speaker_ids,
            'reply_parent': list(reply_parents),
            'reply_relation': list(reply_relations),
            'reply_confidence': list(reply_confidences) if reply_confidences is not None else [1.0] * num_turns,
        }
        if PyGData is None:
            return graph_values
        return PyGData(
            num_nodes=num_turns,
            num_utterance_nodes=torch.tensor([num_turns], dtype=torch.long),
            edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
            edge_type=torch.tensor(edge_types, dtype=torch.long),
            edge_group=torch.tensor(edge_groups, dtype=torch.long),
            edge_weight=torch.tensor(edge_weights, dtype=torch.float),
            speaker_ids_for_turn=torch.tensor(speaker_ids, dtype=torch.long),
        )

    def read_data(self, mode):
        if mode == 'train':
            file_path = self.config.train_path
        elif mode == 'dev':
            file_path = self.config.dev_path
        elif mode == 'test':
            file_path = self.config.test_path
        content = json.load(open(file_path, 'r', encoding='utf-8'))
        res = []
        for line in tqdm(content, desc='Processing dialogues for {}'.format(mode)):
            new_dialog = self.parse_dialogue(line, mode)
            res.append(new_dialog)
        return res

    def parse_dialogue(self, dialogue, mode):
        new_sentences = []
        new_label_sen = []
        target_indices = []
        mask_positions = []
        reply_relations = []
        reply_parents = []
        reply_confidences = []
        target = dialogue["target"]
        knowledge = self.get_knowledge_text(dialogue)
        use_stance_knowledge = bool(getattr(self.config, 'use_knowledge_stance_attention', 1))
        use_knowledge_gate = bool(getattr(self.config, 'use_knowledge_gate', 0))
        knowledge_separate = use_stance_knowledge or use_knowledge_gate
        speakers = dialogue["speakers"]
        raw_sentences = list(dialogue['sentences'])
        local_window = int(getattr(self.config, 'topology_local_window', 3))
        for id, sen in enumerate(raw_sentences):
            previous = raw_sentences[id - 1] if id > 0 else None
            same_speaker = id > 0 and speakers[id] == speakers[id - 1]
            reply_relations.append(self.infer_reply_relation(sen, previous, same_speaker))
            parent, confidence = self.infer_reply_parent_with_confidence(id, speakers)
            reply_parents.append(parent)
            reply_confidences.append(confidence)
            new_sen = (
                f'[CLS]在话语“{sen}”中，用户{speakers[id]}对[SEP]{target}[SEP]的立场为[MASK][SEP]'
                if knowledge_separate
                else f'[CLS]在话语“{sen}”中，用户{speakers[id]}对[SEP]{target}[SEP]的立场为[MASK]。目标说明：{knowledge}[SEP]'
            )
            if id == len(dialogue['sentences']) - 1:
                label_sen = (
                    f'[CLS]在话语“{sen}”中，用户{speakers[id]}对[SEP]{target}[SEP]的立场为{dialogue["label"]}[SEP]'
                    if knowledge_separate
                    else f'[CLS]在话语“{sen}”中，用户{speakers[id]}对[SEP]{target}[SEP]的立场为{dialogue["label"]}。目标说明：{knowledge}[SEP]'
                )
                label_sen_token = self.tokenizer.tokenize(label_sen)
                new_label_sen.append(label_sen_token)
            tokens = self.tokenizer.tokenize(new_sen)
            max_seq_length = int(getattr(self.config, 'max_seq_length', 0))
            if max_seq_length > 0 and len(tokens) > max_seq_length:
                mask_tok = self.tokenizer.mask_token or '[MASK]'
                tokens = self._truncate_tokens_keep_mask(tokens, max_seq_length, mask_token=mask_tok)
            new_sentences.append(tokens)
            mask_positions.append(self.get_mask_position(tokens))
            sep_indices = [i for i, token in enumerate(tokens) if token == '[SEP]']
            target_indices.append([sep_indices[0] + 1, sep_indices[1]])
        dialogue['sentences'] = new_sentences
        dialogue['label_sen'] = new_label_sen
        dialogue['target_idx'] = target_indices
        dialogue['reply_relations'] = reply_relations
        dialogue['reply_parents'] = reply_parents
        dialogue['reply_confidences'] = reply_confidences
        dialogue['topology_graph'] = self.build_topology_graph(
            speakers,
            reply_relations,
            reply_parents,
            reply_confidences=reply_confidences,
            local_window=local_window,
        )
        dialogue['mask_positions'] = mask_positions
        max_knowledge_len = int(getattr(self.config, 'max_knowledge_seq_length', 48))
        if bool(getattr(self.config, 'use_knowledge_stance_attention', 1)) and self.use_target_knowledge:
            favor, against, neutral = self.get_knowledge_stance_texts(dialogue)
            dialogue['knowledge_favor_tokens'] = self._tokenize_knowledge_side(
                self.tokenizer, target, favor, '可能支持', max_knowledge_len,
            )
            dialogue['knowledge_against_tokens'] = self._tokenize_knowledge_side(
                self.tokenizer, target, against, '可能反对', max_knowledge_len,
            )
            dialogue['knowledge_neutral_tokens'] = self._tokenize_knowledge_side(
                self.tokenizer, target, neutral, '背景说明', max_knowledge_len,
            )
        else:
            dialogue['knowledge_favor_tokens'] = []
            dialogue['knowledge_against_tokens'] = []
            dialogue['knowledge_neutral_tokens'] = []
        return dialogue

    def transform2indices(self, data):
        res = []
        for document in data:
            sentences, speakers, label, target_idx, target, label_sen, doc_id, reply_relations, reply_parents, topology_graph, mask_positions, knowledge_favor_tokens, knowledge_against_tokens, knowledge_neutral_tokens = [
                document[w] for w in [
                    'sentences', 'speakers', 'label', 'target_idx', 'target', 'label_sen',
                    'id', 'reply_relations', 'reply_parents', 'topology_graph', 'mask_positions',
                    'knowledge_favor_tokens', 'knowledge_against_tokens', 'knowledge_neutral_tokens',
                ]
            ]
            all_label = document.get('all_label', [label] * len(sentences))
            if len(all_label) != len(sentences):
                raise ValueError(f'id={doc_id} all_label length must match sentences length')
            input_ids = list(map(self.tokenizer.convert_tokens_to_ids, sentences))
            input_masks = [[1] * len(w) for w in input_ids]
            input_segments = [[0] * len(w) for w in input_ids]
            input_ids_label = list(map(self.tokenizer.convert_tokens_to_ids, label_sen))
            input_masks_label = [[1] * len(w) for w in input_ids_label]
            input_segments_label = [[0] * len(w) for w in input_ids_label]
            def encode_side(tokens):
                if tokens:
                    ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    return [ids], [[1] * len(ids)]], [[0] * len(ids)]]
                return [[]], [[]], [[]]

            favor_ids, favor_masks, favor_segments = encode_side(knowledge_favor_tokens)
            against_ids, against_masks, against_segments = encode_side(knowledge_against_tokens)
            neutral_ids, neutral_masks, neutral_segments = encode_side(knowledge_neutral_tokens)

            res.append((
                input_ids, input_masks, input_segments, speakers, label, all_label, target_idx, target,
                reply_relations, reply_parents, topology_graph, input_ids_label, input_masks_label, input_segments_label, doc_id, mask_positions,
                favor_ids, favor_masks, favor_segments,
                against_ids, against_masks, against_segments,
                neutral_ids, neutral_masks, neutral_segments,
            ))
        return res

    def forward(self, mode):
        data = self.read_data(mode)
        res = self.transform2indices(data)
        return res

    def collate_fn_new(self, batch):
        (
            input_ids, input_masks, input_segments, speakers, label, all_label, target_idx, target,
            reply_relations, reply_parents, topology_graphs, input_ids_label, input_masks_label, input_segments_label, doc_id, mask_positions,
            knowledge_favor_ids, knowledge_favor_masks, knowledge_favor_segments,
            knowledge_against_ids, knowledge_against_masks, knowledge_against_segments,
            knowledge_neutral_ids, knowledge_neutral_masks, knowledge_neutral_segments,
        ) = zip(*batch)
        dialogue_length = list(map(len, input_ids))
        st = 0
        dia_idx = []
        for num in dialogue_length:
            dia_idx.append([st, st + num])
            st += num
        max_lens = max(len(w) for sublist in input_ids for w in sublist)
        padding = lambda input_batch: [w + [0] * (max_lens - len(w)) for sublist in input_batch for w in sublist]
        input_ids, input_masks, input_segments = map(padding, [input_ids, input_masks, input_segments])
        max_lens_label = max(len(w) for sublist in input_ids_label for w in sublist)
        padding_label = lambda input_batch: [w + [0] * (max_lens_label - len(w)) for sublist in input_ids_label for w in sublist]
        input_ids_label, input_masks_label, input_segments_label = map(padding_label, [input_ids_label, input_masks_label, input_segments_label])
        def pad_knowledge_batch(batch):
            max_len = max((len(w[0]) for w in batch if w and w[0]), default=0)
            batch_size = len(batch)
            if max_len <= 0:
                return [[0] for _ in range(batch_size)], [[0] for _ in range(batch_size)], [[0] for _ in range(batch_size)]
            ids, masks, segments = [], [], []
            for w in batch:
                seq = w[0] if w and w[0] else []
                pad_len = max_len - len(seq)
                ids.append(seq + [0] * pad_len)
                masks.append([1] * len(seq) + [0] * pad_len)
                segments.append([0] * max_len)
            return ids, masks, segments

        knowledge_favor_ids, knowledge_favor_masks, knowledge_favor_segments = pad_knowledge_batch(knowledge_favor_ids)
        knowledge_against_ids, knowledge_against_masks, knowledge_against_segments = pad_knowledge_batch(knowledge_against_ids)
        knowledge_neutral_ids, knowledge_neutral_masks, knowledge_neutral_segments = pad_knowledge_batch(knowledge_neutral_ids)

        res = {
            "input_ids": torch.tensor(input_ids).to(self.config.device),
            "input_masks": torch.tensor(input_masks).to(self.config.device),
            "input_segments": torch.tensor(input_segments).to(self.config.device),
            "speakers": speakers,
            "label": torch.tensor(label).to(self.config.device),
            "all_label": all_label,
            "dia_idx": dia_idx,
            "target_idx": target_idx,
            "target": target,
            "reply_relations": reply_relations,
            "reply_parents": reply_parents,
            "topology_graphs": topology_graphs,
            "input_ids_label": torch.tensor(input_ids_label).to(self.config.device),
            "input_masks_label": torch.tensor(input_masks_label).to(self.config.device),
            "input_segments_label": torch.tensor(input_segments_label).to(self.config.device),
            "doc_id": doc_id,
            "mask_positions": mask_positions,
            "knowledge_favor_input_ids": torch.tensor(knowledge_favor_ids).to(self.config.device),
            "knowledge_favor_input_masks": torch.tensor(knowledge_favor_masks).to(self.config.device),
            "knowledge_favor_input_segments": torch.tensor(knowledge_favor_segments).to(self.config.device),
            "knowledge_against_input_ids": torch.tensor(knowledge_against_ids).to(self.config.device),
            "knowledge_against_input_masks": torch.tensor(knowledge_against_masks).to(self.config.device),
            "knowledge_against_input_segments": torch.tensor(knowledge_against_segments).to(self.config.device),
            "knowledge_neutral_input_ids": torch.tensor(knowledge_neutral_ids).to(self.config.device),
            "knowledge_neutral_input_masks": torch.tensor(knowledge_neutral_masks).to(self.config.device),
            "knowledge_neutral_input_segments": torch.tensor(knowledge_neutral_segments).to(self.config.device),
        }
        return res

    def get_data(self):
        if self.config.debug:
            train_dataset = MyDataset(self.forward('train'))[:100]
            dev_dataset = MyDataset(self.forward('dev'))[:100]
            test_dataset = MyDataset(self.forward('test'))[:100]
        else:
            train_dataset = MyDataset(self.forward('train'))
            dev_dataset = MyDataset(self.forward('dev'))
            test_dataset = MyDataset(self.forward('test'))
        train_loader = DataLoader(train_dataset, batch_size=self.config.batchsize, shuffle=True, collate_fn=self.collate_fn_new)
        dev_loader = DataLoader(dev_dataset, batch_size=self.config.batchsize, shuffle=False, collate_fn=self.collate_fn_new)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batchsize, shuffle=False, collate_fn=self.collate_fn_new)
        return train_loader, dev_loader, test_loader

