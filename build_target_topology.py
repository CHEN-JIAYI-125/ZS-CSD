import argparse
import json
import os
from collections import defaultdict


def add_node(graph, node_id, node_type, **attrs):
    if node_id not in graph['node_index']:
        graph['node_index'][node_id] = len(graph['nodes'])
        graph['nodes'].append({'id': node_id, 'type': node_type, **attrs})


def add_edge(graph, src, dst, edge_type, **attrs):
    key = (src, dst, edge_type, tuple(sorted(attrs.items())))
    if key in graph['edge_index']:
        return
    graph['edge_index'].add(key)
    graph['edges'].append({'source': src, 'target': dst, 'type': edge_type, **attrs})


def empty_graph(target, target_type):
    return {
        'target': target,
        'target_type': target_type,
        'nodes': [],
        'edges': [],
        'node_index': {},
        'edge_index': set(),
        'stats': defaultdict(int),
    }


def build_topologies(data):
    graphs = {}

    for item in data:
        target = str(item['target'])
        target_type = item.get('target_type', '')
        if target not in graphs:
            graphs[target] = empty_graph(target, target_type)
            add_node(graphs[target], f'target::{target}', 'target', text=target, target_type=target_type)

        graph = graphs[target]
        doc_id = item['id']
        sentences = item['sentences']
        speakers = item['speakers']
        all_label = item.get('all_label', [item['label']] * len(sentences))

        graph['stats']['dialogues'] += 1
        graph['stats']['utterances'] += len(sentences)

        for turn_id, (sentence, speaker, stance) in enumerate(zip(sentences, speakers, all_label)):
            speaker_id = f'target::{target}::speaker::{speaker}'
            add_node(graph, speaker_id, 'speaker', speaker=speaker)

            add_edge(
                graph,
                f'target::{target}',
                speaker_id,
                'utterance-edge',
                doc_id=doc_id,
                turn_id=turn_id,
                speaker=speaker,
                stance=int(stance),
                text=sentence,
            )

    result = []
    for graph in graphs.values():
        graph['stats'] = dict(graph['stats'])
        graph.pop('node_index')
        graph.pop('edge_index')
        result.append(graph)
    return result


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--split', choices=['train', 'dev', 'test', 'all'], default='all')
    parser.add_argument('--output-dir', default='data/topology')
    args = parser.parse_args()

    splits = ['train', 'dev', 'test'] if args.split == 'all' else [args.split]
    for split in splits:
        data_path = os.path.join(args.data_dir, f'{split}_data.json')
        output_path = os.path.join(args.output_dir, f'{split}_target_topology.json')
        graphs = build_topologies(load_json(data_path))
        write_json(output_path, graphs)
        node_count = sum(len(graph['nodes']) for graph in graphs)
        edge_count = sum(len(graph['edges']) for graph in graphs)
        print(f'{split}: {len(graphs)} target graphs, {node_count} nodes, {edge_count} edges -> {output_path}')


if __name__ == '__main__':
    main()
