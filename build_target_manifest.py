"""
Program A: build label-free target manifest.

Only reads target and target_type from split JSON files.
Does not read sentences, speakers, labels, or doc ids.
"""
import argparse
import json
import os

ALLOWED_FIELDS = {'target', 'target_type'}
FORBIDDEN_KEYS = {
    'label',
    'stance',
    'sentences',
    'speakers',
    'doc_id',
    'id',
    'prediction',
}


def extract_manifest_record(item):
    for key in item:
        if key not in ALLOWED_FIELDS and key in FORBIDDEN_KEYS:
            raise ValueError(f'Forbidden key {key!r} in record for target={item.get("target")!r}')
    return {
        'target': str(item['target']),
        'target_type': str(item.get('target_type', '')),
    }


def collect_manifest(data_dir):
    seen = {}
    for split in ('train', 'dev', 'test'):
        path = os.path.join(data_dir, f'{split}_data.json')
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                record = extract_manifest_record(item)
                seen.setdefault(record['target'], record['target_type'])
    manifest = [
        {'target': target, 'target_type': target_type}
        for target, target_type in sorted(seen.items())
    ]
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output', default='data/target_manifest.json')
    args = parser.parse_args()

    manifest = collect_manifest(args.data_dir)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f'Wrote {len(manifest)} targets to {args.output}')


if __name__ == '__main__':
    main()
