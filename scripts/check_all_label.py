"""Quick all_label sanity check on train_data.json."""
import json
from collections import Counter

path = 'data/train_data.json'
with open(path, encoding='utf-8') as f:
    data = json.load(f)

n = len(data)
missing = 0
len_mismatch = 0
final_mismatch = 0
constant_rows = 0
valid_stances = {0, 1, 2}
bad_stance = 0
hist_same_as_final = 0
hist_total = 0
unique_stance_counts = Counter()

for doc in data:
    label = int(doc['label'])
    sents = doc['sentences']
    if 'all_label' not in doc:
        missing += 1
        continue
    al = doc['all_label']
    if len(al) != len(sents):
        len_mismatch += 1
        continue
    if int(al[-1]) != label:
        final_mismatch += 1
    if len(set(al)) == 1:
        constant_rows += 1
    unique_stance_counts[len(set(al))] += 1
    for x in al:
        if int(x) not in valid_stances:
            bad_stance += 1
    for x in al[:-1]:
        hist_total += 1
        if int(x) == label:
            hist_same_as_final += 1

print(f'dialogues: {n}')
print(f'missing all_label: {missing}')
print(f'length mismatch: {len_mismatch}')
print(f'final != all_label[-1]: {final_mismatch}')
print(f'constant [x,x,...,x] rows: {constant_rows} ({100*constant_rows/max(n,1):.1f}%)')
print(f'bad stance values: {bad_stance}')
print(f'hist turn == final label: {hist_same_as_final}/{hist_total} ({100*hist_same_as_final/max(hist_total,1):.1f}%)')
print('unique stance count per dialogue:', dict(sorted(unique_stance_counts.items())))
