import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import RGCNConv
except ImportError:
    RGCNConv = None


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
NUM_EDGE_TYPES = 10


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, nodes, adj):
        hidden = torch.matmul(nodes, self.weight)
        denom = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            output = output + self.bias
        return output


class GLANGlobalBranch(nn.Module):
    def __init__(self, hidden_dim, hop=2, lambdaa=0.5):
        super().__init__()
        self.hop = hop
        self.lambdaa = lambdaa
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, utterances):
        if utterances.size(0) == 1:
            return utterances[-1]
        states = utterances
        length = utterances.size(0)
        mask = torch.zeros(length, 1, device=utterances.device, dtype=utterances.dtype)
        mask[-1] = 1.0
        for hop_idx in range(self.hop):
            alpha = torch.matmul(mask * states, utterances.transpose(0, 1))
            if hop_idx == self.hop - 1:
                return states[-1]
            attended = alpha.transpose(0, 1)[:, length - 1:length] * utterances
            states = self.lambdaa * self.norm(torch.sigmoid(attended)) + utterances
        return states[-1]


class GLANLocalBranch(nn.Module):
    def __init__(self, hidden_dim, hop=2, lambdaa=0.5):
        super().__init__()
        self.hop = hop
        self.lambdaa = lambdaa
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, utterances):
        if utterances.size(0) == 0:
            raise ValueError('local branch requires at least one utterance')
        if utterances.size(0) == 1:
            return utterances[-1]
        conv_out = torch.tanh(self.conv1(utterances.transpose(0, 1).unsqueeze(0)))
        conv_out = torch.tanh(self.conv2(conv_out)).squeeze(0).transpose(0, 1)
        length = conv_out.size(0)
        mask = torch.zeros(length, 1, device=utterances.device, dtype=utterances.dtype)
        mask[-1] = 1.0
        for hop_idx in range(self.hop):
            alpha = torch.matmul(mask * conv_out, utterances.transpose(0, 1))
            if hop_idx == self.hop - 1:
                return conv_out[-1]
            attended = alpha.transpose(0, 1)[:, length - 1:length] * utterances
            conv_out = self.lambdaa * self.norm(torch.sigmoid(attended)) + utterances
        return conv_out[-1]


class GLANStructuralBranch(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, num_relations=NUM_EDGE_TYPES, dropout=0.1, hop=2, lambdaa=0.5):
        super().__init__()
        self.hop = hop
        self.lambdaa = lambdaa
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gc1 = GraphConvolution(hidden_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.use_rgcn = RGCNConv is not None
        if self.use_rgcn:
            self.rgcn1 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
            self.rgcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        else:
            self.rgcn1 = None
            self.rgcn2 = None

    def build_adjacency(self, graph, num_nodes, device, dtype):
        adj = torch.eye(num_nodes, device=device, dtype=dtype)
        if graph is None:
            return adj
        edge_index = _graph_tensor(graph, 'edge_index')
        edge_weight = _graph_tensor(graph, 'edge_weight')
        if edge_index is None or edge_index.numel() == 0:
            return adj
        num_utterances = int(_graph_scalar(graph, 'num_utterance_nodes', num_nodes))
        src = edge_index[0]
        dst = edge_index[1]
        weights = edge_weight if edge_weight is not None else torch.ones(src.size(0), device=device, dtype=dtype)
        for s, d, w in zip(src.tolist(), dst.tolist(), weights.tolist()):
            if 0 <= s < num_utterances and 0 <= d < num_utterances:
                adj[d, s] = adj[d, s] + w
        return adj

    def forward(self, utterances, graph):
        length = utterances.size(0)
        if length == 1:
            return utterances[-1], utterances
        adj = self.build_adjacency(graph, length, utterances.device, utterances.dtype)
        adj = adj[:length, :length]
        gcn_1 = torch.tanh(self.gc1(utterances, adj))
        gcn_out = torch.tanh(self.gc2(gcn_1, adj))
        mask = torch.zeros(length, 1, device=utterances.device, dtype=utterances.dtype)
        mask[-1] = 1.0
        states = gcn_out
        for hop_idx in range(self.hop):
            alpha = torch.matmul(mask * states, utterances.transpose(0, 1))
            if hop_idx == self.hop - 1:
                return states[-1], states
            attended = alpha.transpose(0, 1)[:, length - 1:length] * utterances
            states = self.lambdaa * self.norm(torch.sigmoid(attended)) + utterances
        return states[-1], states

    def forward_rgcn(self, node_features, graph):
        if not self.use_rgcn or graph is None:
            return node_features
        edge_index = _graph_tensor(graph, 'edge_index')
        edge_type = _graph_tensor(graph, 'edge_type')
        if edge_index is None or edge_index.numel() == 0:
            return node_features
        device = node_features.device
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
        x = node_features
        x = torch.tanh(self.rgcn1(x, edge_index, edge_type))
        x = self.dropout(x)
        x = torch.tanh(self.rgcn2(x, edge_index, edge_type))
        return x


class TargetTopologyEncoder(nn.Module):
    """GLAN-style global / local / structural branches with target-attention fusion."""

    def __init__(
        self,
        hidden_dim,
        num_layers=2,
        dropout=0.1,
        local_window=3,
        hop=2,
        lambdaa=0.5,
        branches=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.local_window = local_window
        self.branches = set(branches or ['global', 'local', 'struct', 'target'])
        self.global_branch = GLANGlobalBranch(hidden_dim, hop=hop, lambdaa=lambdaa)
        self.local_branch = GLANLocalBranch(hidden_dim, hop=hop, lambdaa=lambdaa)
        self.structural_branch = GLANStructuralBranch(
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            hop=hop,
            lambdaa=lambdaa,
        )
        self.branch_proj = nn.ModuleDict({
            name: nn.Linear(hidden_dim, hidden_dim)
            for name in ['global', 'local', 'struct', 'final']
        })
        self.target_query = nn.Linear(hidden_dim, hidden_dim)
        self.target_key = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _local_slice(self, utterances, graph):
        if graph is not None and hasattr(graph, 'local_window'):
            start = int(graph.local_window[0].item()) if torch.is_tensor(graph.local_window) else int(graph.local_window[0])
        elif graph is not None and isinstance(graph, dict) and 'local_window' in graph:
            start = int(graph['local_window'][0])
        else:
            start = max(0, utterances.size(0) - self.local_window)
        return utterances[start:]

    def _build_node_features(self, utterances, target_repr, graph):
        length = utterances.size(0)
        device = utterances.device
        dtype = utterances.dtype
        if graph is None:
            return utterances, length, 0

        num_speakers = int(_graph_scalar(graph, 'num_speaker_nodes', 0))
        num_nodes = int(_graph_scalar(graph, 'num_nodes', length + num_speakers + 1))
        speaker_offset = int(_graph_scalar(graph, 'speaker_node_offset', length))
        target_node = int(_graph_scalar(graph, 'target_node', num_nodes - 1))
        speaker_for_turn = _graph_tensor(graph, 'speaker_ids_for_turn')
        x = torch.zeros(num_nodes, self.hidden_dim, device=device, dtype=dtype)
        x[:length] = utterances
        if speaker_for_turn is not None and num_speakers > 0:
            for turn_id in range(length):
                speaker_local = int(speaker_for_turn[turn_id].item())
                x[speaker_offset + speaker_local] = x[speaker_offset + speaker_local] + utterances[turn_id]
            for speaker_local in range(num_speakers):
                node_id = speaker_offset + speaker_local
                turns = (speaker_for_turn == speaker_local).nonzero(as_tuple=False).view(-1)
                if turns.numel() > 0:
                    x[node_id] = utterances[turns].mean(dim=0)
        if 0 <= target_node < num_nodes:
            x[target_node] = target_repr
        return x, length, speaker_offset

    def forward(self, utterances, target_repr, graph=None, edge_mode='utterance'):
        length = utterances.size(0)
        branch_outputs = []
        branch_names = []

        if 'global' in self.branches:
            branch_outputs.append(self.branch_proj['global'](self.global_branch(utterances)))
            branch_names.append('global')
        if 'local' in self.branches:
            branch_outputs.append(self.branch_proj['local'](self.local_branch(self._local_slice(utterances, graph))))
            branch_names.append('local')
        if 'struct' in self.branches:
            h_struct, graph_v = self.structural_branch(utterances, graph)
            node_x, utt_count, _ = self._build_node_features(utterances, target_repr, graph)
            if self.structural_branch.use_rgcn and graph is not None and _graph_scalar(graph, 'num_nodes', 0) > utt_count:
                node_x = self.structural_branch.forward_rgcn(node_x, graph)
                graph_v = node_x[:utt_count]
                h_struct = graph_v[-1]
            branch_outputs.append(self.branch_proj['struct'](h_struct))
            branch_names.append('struct')
        else:
            _, graph_v = self.structural_branch(utterances, graph)

        if 'final' in self.branches:
            branch_outputs.append(self.branch_proj['final'](utterances[-1]))
            branch_names.append('final')

        if not branch_outputs:
            target_state = utterances[-1]
            stacked = None
        else:
            stacked = torch.stack(branch_outputs, dim=0)
            query = self.target_query(target_repr).unsqueeze(0)
            keys = self.target_key(stacked)
            scores = torch.matmul(query, keys.transpose(0, 1)).squeeze(0) / (self.hidden_dim ** 0.5)
            weights = torch.softmax(scores, dim=0).view(-1, 1)
            target_state = (weights * stacked).sum(dim=0)
            target_state = self.fusion_norm(target_state)

        target_state = self.dropout(target_state)
        if graph_v.size(0) != length:
            graph_v = utterances
        return graph_v, target_state, branch_names, stacked


def _graph_tensor(graph, name):
    if graph is None:
        return None
    if hasattr(graph, name):
        value = getattr(graph, name)
        return value if torch.is_tensor(value) else torch.tensor(value)
    if isinstance(graph, dict) and name in graph:
        value = graph[name]
        return value if torch.is_tensor(value) else torch.tensor(value)
    return None


def _graph_scalar(graph, name, default=0):
    if graph is None:
        return default
    if hasattr(graph, name):
        value = getattr(graph, name)
    elif isinstance(graph, dict) and name in graph:
        value = graph[name]
    else:
        return default
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def stance_relation_loss(utterances, labels, config):
    labels = torch.tensor(labels, dtype=torch.long, device=utterances.device)
    length = utterances.size(0)
    if length < 2:
        return torch.zeros((), device=utterances.device)

    tau = float(getattr(config, 'topology_tau', 0.2))
    max_pairs = int(getattr(config, 'topology_max_pairs', 128))
    neutral_weight = float(getattr(config, 'topology_neutral_weight', 0.5))
    fa_weight = float(getattr(config, 'topology_favor_against_weight', 1.5))

    utterances = F.normalize(utterances, p=2, dim=-1)
    losses = []
    pair_count = 0
    for i in range(length):
        for j in range(i + 1, length):
            if pair_count >= max_pairs:
                break
            sim = torch.dot(utterances[i], utterances[j]) / tau
            if labels[i] == labels[j]:
                if labels[i].item() == 2:
                    weight = neutral_weight
                else:
                    weight = fa_weight
                losses.append(weight * (1.0 - sim).pow(2))
            else:
                losses.append(F.relu(sim).pow(2))
            pair_count += 1
        if pair_count >= max_pairs:
            break
    if not losses:
        return torch.zeros((), device=utterances.device)
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Context topology encoder (v3): reply/next-turn graph; speaker handled in SSE hypergraph
# ---------------------------------------------------------------------------

EDGE_GROUP_CONTEXT = 0
EDGE_GROUP_SPEAKER_HISTORY = 1
EDGE_GROUP_AUXILIARY = 2


def summarize_edge_groups(graph):
    """Return edge counts per group; use for debugging graph construction."""
    edge_group = _graph_tensor(graph, 'edge_group')
    if edge_group is None:
        return {
            'context': 0,
            'speaker': 0,
            'auxiliary': 0,
            'missing_edge_group': True,
        }
    counts = torch.bincount(edge_group.long().view(-1), minlength=3)
    return {
        'context': int(counts[0].item()),
        'speaker': int(counts[1].item()),
        'auxiliary': int(counts[2].item()),
        'missing_edge_group': False,
    }


def build_group_adjacency(graph, num_nodes, group_id, device, dtype):
    """Row-normalized weighted adjacency for one edge group (src -> dst)."""
    adj = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)
    edge_index = _graph_tensor(graph, 'edge_index')
    edge_group = _graph_tensor(graph, 'edge_group')
    edge_weight = _graph_tensor(graph, 'edge_weight')
    if edge_index is None or edge_index.numel() == 0:
        return adj
    if edge_group is None:
        raise ValueError('topology graph is missing edge_group; rebuild with dataset_v3')

    if edge_index.device != device:
        edge_index = edge_index.to(device)
    src = edge_index[0].long()
    dst = edge_index[1].long()
    groups = edge_group.long().to(device)
    weights = edge_weight.to(device=device, dtype=dtype) if edge_weight is not None else torch.ones(src.size(0), device=device, dtype=dtype)

    for i in range(src.size(0)):
        if int(groups[i].item()) != group_id:
            continue
        s = int(src[i].item())
        d = int(dst[i].item())
        if s == d or not (0 <= s < num_nodes and 0 <= d < num_nodes):
            continue
        adj[d, s] = adj[d, s] + weights[i]

    row_sum = adj.sum(dim=1, keepdim=True).clamp_min(1e-6)
    adj = adj / row_sum
    return adj


class SpeakerHypergraphChannel(nn.Module):
    """
    Speaker-dimension hyperedges: for each turn, pool all same-speaker
    historical utterances (j <= i) with target-aware attention.
    """

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_dim ** -0.5

    def forward(self, nodes, speaker_ids_for_turn):
        num_nodes = nodes.size(0)
        device = nodes.device
        encoded = F.gelu(self.linear(nodes))
        encoded = self.dropout(encoded)

        if speaker_ids_for_turn is None:
            return encoded

        speaker_ids = speaker_ids_for_turn.to(device).long().view(-1)
        if speaker_ids.numel() != num_nodes:
            return encoded

        messages = []
        for turn_id in range(num_nodes):
            spk = speaker_ids[turn_id]
            hist_mask = (speaker_ids == spk) & (
                torch.arange(num_nodes, device=device) <= turn_id
            )
            hist_idx = hist_mask.nonzero(as_tuple=False).view(-1)
            if hist_idx.numel() == 0:
                messages.append(encoded[turn_id])
                continue
            hist = encoded[hist_idx]
            query = encoded[turn_id:turn_id + 1]
            scores = torch.matmul(query, hist.transpose(0, 1)) * self.scale
            weights = torch.softmax(scores, dim=-1)
            messages.append(torch.matmul(weights, hist).squeeze(0))
        return torch.stack(messages)


class ContextTopologyEncoder(nn.Module):
    """
    Context graph propagation; speaker history is handled inside SSE hypergraph.
    Normalize node + context message, concat, project back to hidden_dim.
    """

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.context_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.channel_merge = nn.Linear(hidden_dim * 2, hidden_dim)
        self.message_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def channel_weights(self):
        return None

    def forward(self, nodes, graph):
        num_nodes = nodes.size(0)
        device = nodes.device
        dtype = nodes.dtype

        adj_context = build_group_adjacency(graph, num_nodes, EDGE_GROUP_CONTEXT, device, dtype)
        context_message = F.gelu(torch.matmul(adj_context, self.context_linear(nodes)))
        context_message = self.message_dropout(context_message)

        merged = torch.cat([
            self.node_norm(nodes),
            self.context_norm(context_message),
        ], dim=-1)
        return self.norm(self.channel_merge(merged))
