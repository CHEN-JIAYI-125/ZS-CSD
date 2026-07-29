"""
Compress auditable web-full target knowledge into model-facing cards.

Input:  data/target_knowledge_web_full.json
Output: data/target_knowledge_model.json

The compressor is deterministic and does not read train/dev/test dialogues or labels.
"""
import argparse
import json
import os

from src.dataset.dataset_v3 import DataProcessor

FORBIDDEN_TOP_LEVEL = {'label', 'stance', 'sentences', 'speakers', 'doc_id', 'prediction'}


def audit_full_record(target, record):
    if not isinstance(record, dict):
        return
    for key in FORBIDDEN_TOP_LEVEL:
        if key in record:
            raise ValueError(f'Forbidden key {key!r} in full knowledge for {target!r}')


def compress_file(full_path, output_path, max_chars):
    with open(full_path, 'r', encoding='utf-8') as f:
        full = json.load(f)

    model_cards = {}
    for target, entry in full.items():
        audit_full_record(target, entry)
        fields = DataProcessor.normalize_full_entry_to_model(entry)
        model_cards[str(target)] = {
            field: fields[field]
            for field in DataProcessor.MODEL_KNOWLEDGE_FIELDS
        }
        # also store compressed prompt string for inspection
        model_cards[str(target)]['_compressed_preview'] = DataProcessor.compress_knowledge_card(
            fields,
            max_total=max_chars,
        )

    # strip preview keys from model export if user wants strict 4-field only
    export = {}
    for target, card in model_cards.items():
        export[target] = {k: v for k, v in card.items() if not k.startswith('_')}

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f'Wrote {len(export)} model cards to {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/target_knowledge_web_full.json')
    parser.add_argument('--output', default='data/target_knowledge_model.json')
    parser.add_argument('--max-chars', type=int, default=200)
    args = parser.parse_args()
    compress_file(args.input, args.output, args.max_chars)


if __name__ == '__main__':
    main()
