"""Compute Table 2 metrics (F/A/N/M) from pred CSV + dataset JSON."""
import argparse
import json
from pathlib import Path

from sklearn.metrics import f1_score


def load_target_type_map(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return {item['id']: item.get('target_type', '') for item in data}


def subset_metrics(trues, preds):
    if not trues:
        return {'favor': float('nan'), 'against': float('nan'), 'neutral': float('nan'), 'macro': float('nan'), 'count': 0}
    macro = f1_score(trues, preds, average='macro')
    favor, against, neutral = f1_score(trues, preds, average=None)
    return {
        'favor': favor,
        'against': against,
        'neutral': neutral,
        'macro': macro,
        'count': len(trues),
    }


def pct(value):
    return 'NA' if value != value else f'{100 * value:.2f}'


def format_row(label, metrics):
    return (
        f'{label:<20} | '
        f'F: {pct(metrics["favor"]):>6} | '
        f'A: {pct(metrics["against"]):>6} | '
        f'N: {pct(metrics["neutral"]):>6} | '
        f'M: {pct(metrics["macro"]):>6} | '
        f'n={metrics["count"]}'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help='pred CSV with columns doc_id,true,pred')
    parser.add_argument('--test', default='data/test_data.json')
    parser.add_argument('--out', default='', help='optional output txt path')
    args = parser.parse_args()

    pred_df = __import__('pandas').read_csv(args.pred)
    target_types = load_target_type_map(args.test)

    trues = pred_df['true'].tolist()
    preds = pred_df['pred'].tolist()
    doc_ids = pred_df['doc_id'].tolist()

    grouped = {'n': ([], []), 'c': ([], [])}
    for true, pred, doc_id in zip(trues, preds, doc_ids):
        ttype = target_types.get(doc_id, '')
        if ttype in grouped:
            grouped[ttype][0].append(true)
            grouped[ttype][1].append(pred)

    metrics = {
        'mixed': subset_metrics(trues, preds),
        'noun_phrase': subset_metrics(*grouped['n']),
        'claim': subset_metrics(*grouped['c']),
    }

    lines = [
        'Table 2 metrics from pred CSV',
        format_row('Noun-phrase targets', metrics['noun_phrase']),
        format_row('Claim targets', metrics['claim']),
        format_row('Mixed targets', metrics['mixed']),
        '',
        f'{"Group":<20} {"F":>8} {"A":>8} {"N":>8} {"M":>8} {"Count":>8}',
    ]
    for label, key in [('Noun-phrase', 'noun_phrase'), ('Claim', 'claim'), ('Mixed', 'mixed')]:
        m = metrics[key]
        lines.append(
            f'{label:<20} {pct(m["favor"]):>8} {pct(m["against"]):>8} '
            f'{pct(m["neutral"]):>8} {pct(m["macro"]):>8} {m["count"]:>8}'
        )

    text = '\n'.join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + '\n', encoding='utf-8')
        print(f'\nSaved to {args.out}')


if __name__ == '__main__':
    main()
