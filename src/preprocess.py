from transformers import AutoTokenizer
import json
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

# Backward-compatible aliases used by relation-context heuristics.
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


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class DataProcessor():
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_dir)
        self.config = config
        self.target_knowledge = self.load_target_knowledge()

    def load_target_knowledge(self):
        path = getattr(self.config, 'target_knowledge_path', '')
        if not path or not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        knowledge = {}
        for target, value in raw.items():
            if isinstance(value, str):
                knowledge[str(target)] = value.strip()
            elif isinstance(value, dict):
                parts = []
                for key in ['description', 'favor_reason', 'against_reason', 'neutral_hint']:
                    text = str(value.get(key, '')).strip()
                    if text:
                        parts.append(f'{key}: {text}')
                knowledge[str(target)] = ' '.join(parts)
        return knowledge

    def get_knowledge_text(self, dialogue):
        target = str(dialogue['target'])
        text = self.target_knowledge.get(target, '')
        if not text:
            text = f'target={target}; target_type={dialogue.get("target_type", "")}'
        max_chars = int(getattr(self.config, 'target_knowledge_max_chars', 120))
        return text[:max_chars] if max_chars > 0 else text

    def get_mask_position(self, tokens):
        mask_token = self.tokenizer.mask_token or '[MASK]'
        if mask_token in tokens:
            return tokens.index(mask_token)
        if '[MASK]' in tokens:
            return tokens.index('[MASK]')
        raise ValueError('Cannot find [MASK] token in tokenized sentence')

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

    def build_topology_graph(self, speakers, reply_relations, reply_parents, reply_confidences=None, local_window=3):
        """Build utterance-centered GLAN-style topology graph."""
        speaker_ids = [int(speaker) for speaker in speakers]
        num_turns = len(speaker_ids)
        speaker_to_local = {}
        speaker_ids_for_turn = []
        for speaker in speaker_ids:
            if speaker not in speaker_to_local:
                speaker_to_local[speaker] = len(speaker_to_local)
            speaker_ids_for_turn.append(speaker_to_local[speaker])

        num_speaker_nodes = len(speaker_to_local)
        speaker_node_offset = num_turns
        target_node = num_turns + num_speaker_nodes
        num_nodes = target_node + 1

        edge_src = []
        edge_dst = []
        edge_types = []
        edge_weights = []
        edge_turn_indices = []
        last_turn_by_speaker = {}

        def add_edge(src, dst, edge_type, weight=1.0, turn_idx=-1):
            edge_src.append(src)
            edge_dst.append(dst)
            edge_types.append(edge_type)
            edge_weights.append(float(weight))
            edge_turn_indices.append(turn_idx)

        for turn_id, speaker in enumerate(speaker_ids):
            current_node = turn_id
            add_edge(current_node, current_node, EDGE_SELF, 1.0, turn_id)

            if turn_id > 0:
                add_edge(turn_id - 1, turn_id, EDGE_NEXT_TURN, 1.0, turn_id)

            parent = reply_parents[turn_id] if turn_id < len(reply_parents) else -1
            confidence = reply_confidences[turn_id] if reply_confidences is not None else 1.0
            if 0 <= parent < turn_id:
                relation = reply_relations[turn_id] if turn_id < len(reply_relations) else RELATION_CROSS_REPLY
                edge_type = self.reply_relation_to_edge_type(relation, speaker_ids[parent] == speaker)
                add_edge(parent, turn_id, edge_type, confidence, turn_id)
            elif turn_id == 0:
                add_edge(target_node, turn_id, EDGE_ROOT_TO_UTT, 0.45, turn_id)

            if speaker in last_turn_by_speaker:
                add_edge(last_turn_by_speaker[speaker], turn_id, EDGE_SAME_SPEAKER, 1.0, turn_id)
            last_turn_by_speaker[speaker] = turn_id

            speaker_node = speaker_node_offset + speaker_to_local[speaker]
            add_edge(speaker_node, turn_id, EDGE_SPEAKER_TO_UTT, 1.0, turn_id)
            add_edge(target_node, turn_id, EDGE_TARGET_TO_UTT, 1.0, turn_id)

        hyper_nodes = []
        hyper_edges = []
        hyper_turn_indices = []
        hyper_position_ids = []
        hyperedge_count = 0
        for turn_id in range(num_turns):
            chain = self.reply_chain(turn_id, reply_parents)
            if not chain:
                continue
            hyperedge_id = hyperedge_count
            hyperedge_count += 1
            for position, chain_turn in enumerate(chain):
                hyper_nodes.append(chain_turn)
                hyper_edges.append(hyperedge_id)
                hyper_turn_indices.append(chain_turn)
                distance_to_current = min(len(chain) - position - 1, 7)
                hyper_position_ids.append(distance_to_current)

        local_start = max(0, num_turns - int(local_window))
        local_window_range = [local_start, num_turns]
        graph_values = {
            'num_utterance_nodes': num_turns,
            'num_speaker_nodes': num_speaker_nodes,
            'num_nodes': num_nodes,
            'speaker_node_offset': speaker_node_offset,
            'target_node': target_node,
            'speaker_ids_for_turn': speaker_ids_for_turn,
            'edge_index': [edge_src, edge_dst],
            'edge_type': edge_types,
            'edge_types': edge_types,
            'edge_weight': edge_weights,
            'edge_turn_index': edge_turn_indices,
            'edge_turn_indices': edge_turn_indices,
            'hyperedge_index': [hyper_nodes, hyper_edges],
            'hyperedge_turn_index': hyper_turn_indices,
            'hyperedge_turn_indices': hyper_turn_indices,
            'hyperedge_position_id': hyper_position_ids,
            'hyperedge_position_ids': hyper_position_ids,
            'num_hyperedges': hyperedge_count,
            'local_window': local_window_range,
            'reply_parent': list(reply_parents),
            'reply_relation': list(reply_relations),
            'reply_confidence': list(reply_confidences) if reply_confidences is not None else [1.0] * num_turns,
        }
        if PyGData is None:
            return graph_values
        return PyGData(
            num_nodes=num_nodes,
            num_utterance_nodes=torch.tensor([num_turns], dtype=torch.long),
            num_speaker_nodes=torch.tensor([num_speaker_nodes], dtype=torch.long),
            speaker_node_offset=torch.tensor([speaker_node_offset], dtype=torch.long),
            target_node=torch.tensor([target_node], dtype=torch.long),
            speaker_ids_for_turn=torch.tensor(speaker_ids_for_turn, dtype=torch.long),
            edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
            edge_type=torch.tensor(edge_types, dtype=torch.long),
            edge_weight=torch.tensor(edge_weights, dtype=torch.float),
            edge_turn_index=torch.tensor(edge_turn_indices, dtype=torch.long),
            hyperedge_index=torch.tensor([hyper_nodes, hyper_edges], dtype=torch.long) if hyper_nodes else torch.empty((2, 0), dtype=torch.long),
            hyperedge_turn_index=torch.tensor(hyper_turn_indices, dtype=torch.long) if hyper_turn_indices else torch.empty((0,), dtype=torch.long),
            hyperedge_position_id=torch.tensor(hyper_position_ids, dtype=torch.long) if hyper_position_ids else torch.empty((0,), dtype=torch.long),
            num_hyperedges=torch.tensor([hyperedge_count], dtype=torch.long),
            local_window=torch.tensor(local_window_range, dtype=torch.long),
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
            new_sen = f'[CLS]在话语“{sen}”中，用户{speakers[id]}对[SEP]{target}[SEP]的立场为[MASK]。目标说明：{knowledge}[SEP]'
            if id == len(dialogue['sentences']) - 1:
                label_sen = f'[CLS]在话语“{sen}”中，用户{speakers[id]}对[SEP]{target}[SEP]的立场为{dialogue["label"]}。目标说明：{knowledge}[SEP]'
                label_sen_token = self.tokenizer.tokenize(label_sen)
                new_label_sen.append(label_sen_token)
            tokens = self.tokenizer.tokenize(new_sen)
            max_seq_length = int(getattr(self.config, 'max_seq_length', 0))
            if max_seq_length > 0 and len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length - 1] + ['[SEP]']
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
        return dialogue

    def transform2indices(self, data):
        res = []
        for document in data:
            sentences, speakers, label, target_idx, target, label_sen, doc_id, reply_relations, reply_parents, topology_graph, mask_positions = [
                document[w] for w in [
                    'sentences', 'speakers', 'label', 'target_idx', 'target', 'label_sen',
                    'id', 'reply_relations', 'reply_parents', 'topology_graph', 'mask_positions'
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
            res.append((
                input_ids, input_masks, input_segments, speakers, label, all_label, target_idx, target,
                reply_relations, reply_parents, topology_graph, input_ids_label, input_masks_label, input_segments_label, doc_id, mask_positions
            ))
        return res

    def forward(self, mode):
        data = self.read_data(mode)
        res = self.transform2indices(data)
        return res

    def collate_fn_new(self, batch):
        (
            input_ids, input_masks, input_segments, speakers, label, all_label, target_idx, target,
            reply_relations, reply_parents, topology_graphs, input_ids_label, input_masks_label, input_segments_label, doc_id, mask_positions
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
            "mask_positions": mask_positions
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

