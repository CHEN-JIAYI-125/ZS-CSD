import argparse
import json
import os


def collect_targets(data_dir):
    targets = {}
    for split in ['train', 'dev', 'test']:
        path = os.path.join(data_dir, f'{split}_data.json')
        with open(path, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                target = str(item['target'])
                targets.setdefault(target, item.get('target_type', ''))
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output', default='data/target_knowledge_template.json')
    args = parser.parse_args()

    targets = collect_targets(args.data_dir)
    template = {}
    for target, target_type in sorted(targets.items()):
        template[target] = {
            'description': '',
            'favor_reason': '',
            'against_reason': '',
            'neutral_hint': '',
            'target_type': target_type
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    print(f'wrote {len(template)} targets to {args.output}')


if __name__ == '__main__':
    main()
