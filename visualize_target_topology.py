import argparse
import html
import json
import math
import os


NODE_COLORS = {
    'target': '#5b2be0',
    'speaker': '#2477b3',
}

EDGE_COLORS = {
    'utterance_0': '#2e9d57',
    'utterance_1': '#d64b3c',
    'utterance_2': '#8a8f98',
}

STANCE_NAMES = {
    0: 'favor',
    1: 'against',
    2: 'neutral',
}


def load_graphs(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def pick_graph(graphs, target):
    if target:
        for graph in graphs:
            if graph['target'] == target:
                return graph
        raise ValueError(f'target not found: {target}')
    return max(graphs, key=lambda graph: graph.get('stats', {}).get('utterances', 0))


def trim_graph(graph, max_utterances):
    target_nodes = [node for node in graph['nodes'] if node['type'] == 'target']
    utterance_edges = [edge for edge in graph['edges'] if edge['type'] == 'utterance-edge']
    utterance_edges = utterance_edges[:max_utterances]
    speaker_ids = {edge['target'] for edge in utterance_edges}

    nodes = target_nodes + [
        node for node in graph['nodes']
        if node['type'] == 'speaker' and node['id'] in speaker_ids
    ]
    node_ids = {node['id'] for node in nodes}
    edges = [
        edge for edge in utterance_edges
        if edge['source'] in node_ids and edge['target'] in node_ids
    ]
    return nodes, edges


def point_on_circle(cx, cy, radius, index, total, start_angle=-math.pi / 2):
    if total <= 0:
        return cx, cy
    angle = start_angle + 2 * math.pi * index / total
    return cx + radius * math.cos(angle), cy + radius * math.sin(angle)


def layout_nodes(nodes, width, height):
    target_nodes = [node for node in nodes if node['type'] == 'target']
    speaker_nodes = [node for node in nodes if node['type'] == 'speaker']

    cx, cy = width / 2, height / 2
    positions = {}
    if target_nodes:
        positions[target_nodes[0]['id']] = (cx, cy)

    speaker_radius = min(width, height) * 0.22
    for idx, node in enumerate(speaker_nodes):
        positions[node['id']] = point_on_circle(cx, cy, speaker_radius, idx, len(speaker_nodes))

    return positions


def node_color(node):
    return NODE_COLORS[node['type']]


def node_radius(node):
    if node['type'] == 'target':
        return 18
    return 11


def node_label(node):
    if node['type'] == 'target':
        return str(node.get('text', node['id']))
    return f"S{node.get('speaker')}"


def node_tooltip(node):
    return json.dumps(node, ensure_ascii=False)


def edge_tooltip(edge):
    stance = STANCE_NAMES.get(int(edge.get('stance', 2)), edge.get('stance'))
    text = str(edge.get('text', ''))
    return (
        f"type={edge.get('type')} doc={edge.get('doc_id')} turn={edge.get('turn_id')} "
        f"speaker={edge.get('speaker')} stance={stance}\n{text}"
    )


def write_html(graph, nodes, edges, output, width=1400, height=1000):
    positions = layout_nodes(nodes, width, height)
    node_by_id = {node['id']: node for node in nodes}
    lines = []
    lines.append('<!doctype html><html><head><meta charset="utf-8">')
    lines.append('<title>Target Topology</title>')
    lines.append('''<style>
body { font-family: Arial, sans-serif; margin: 0; background: #fafafa; color: #202124; }
.wrap { padding: 18px 24px; }
.meta { margin-bottom: 12px; line-height: 1.5; }
.legend span { display: inline-flex; align-items: center; margin-right: 18px; font-size: 13px; }
.dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 6px; }
svg { width: 100%; height: auto; background: #fff; border: 1px solid #ddd; }
.edge { stroke-opacity: 0.34; }
.node-label { font-size: 11px; fill: #111; pointer-events: none; text-anchor: middle; }
</style>''')
    lines.append('</head><body><div class="wrap">')
    stats = graph.get('stats', {})
    lines.append(
        f'<div class="meta"><b>Target:</b> {html.escape(str(graph["target"]))} '
        f'&nbsp; <b>type:</b> {html.escape(str(graph.get("target_type", "")))} '
        f'&nbsp; <b>shown nodes:</b> {len(nodes)} '
        f'&nbsp; <b>shown edges:</b> {len(edges)} '
        f'&nbsp; <b>full utterances:</b> {stats.get("utterances", "-")}</div>'
    )
    lines.append('<div class="legend">')
    lines.append(f'<span><i class="dot" style="background:{NODE_COLORS["target"]}"></i>target</span>')
    lines.append(f'<span><i class="dot" style="background:{NODE_COLORS["speaker"]}"></i>speaker</span>')
    lines.append(f'<span><i class="dot" style="background:{EDGE_COLORS["utterance_0"]}"></i>favor utterance edge</span>')
    lines.append(f'<span><i class="dot" style="background:{EDGE_COLORS["utterance_1"]}"></i>against utterance edge</span>')
    lines.append(f'<span><i class="dot" style="background:{EDGE_COLORS["utterance_2"]}"></i>neutral utterance edge</span>')
    lines.append('</div>')
    lines.append(f'<svg viewBox="0 0 {width} {height}" role="img">')

    for edge in edges:
        if edge['source'] not in positions or edge['target'] not in positions:
            continue
        x1, y1 = positions[edge['source']]
        x2, y2 = positions[edge['target']]
        color = EDGE_COLORS.get(f'utterance_{edge.get("stance")}', EDGE_COLORS['utterance_2'])
        stroke_width = 1.4
        title = html.escape(edge_tooltip(edge))
        lines.append(
            f'<line class="edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{stroke_width}"><title>{title}</title></line>'
        )

    for node in nodes:
        x, y = positions[node['id']]
        color = node_color(node)
        radius = node_radius(node)
        title = html.escape(node_tooltip(node))
        label = html.escape(node_label(node))
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" stroke="#fff" stroke-width="2">'
            f'<title>{title}</title></circle>'
        )
        label_y = y + radius + 12
        lines.append(f'<text class="node-label" x="{x:.1f}" y="{label_y:.1f}">{label}</text>')

    lines.append('</svg>')
    lines.append('</div></body></html>')

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def list_targets(graphs, limit):
    rows = sorted(
        ((graph['target'], graph.get('target_type', ''), graph.get('stats', {}).get('utterances', 0)) for graph in graphs),
        key=lambda row: (-row[2], row[0])
    )
    for target, target_type, utterances in rows[:limit]:
        print(f'{target}\t{target_type}\t{utterances}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topology', default='data/topology/train_target_topology.json')
    parser.add_argument('--target', default='')
    parser.add_argument('--output', default='result/topology_view.html')
    parser.add_argument('--max-utterances', type=int, default=80)
    parser.add_argument('--list-targets', action='store_true')
    parser.add_argument('--list-limit', type=int, default=30)
    args = parser.parse_args()

    graphs = load_graphs(args.topology)
    if args.list_targets:
        list_targets(graphs, args.list_limit)
        return

    graph = pick_graph(graphs, args.target)
    nodes, edges = trim_graph(graph, args.max_utterances)
    write_html(graph, nodes, edges, args.output)
    print(f'wrote {args.output}')
    print(f'target={graph["target"]}, shown_nodes={len(nodes)}, shown_edges={len(edges)}')


if __name__ == '__main__':
    main()
