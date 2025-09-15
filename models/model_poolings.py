import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv, GINEConv, BatchNorm,
    global_mean_pool, global_max_pool, global_add_pool, GlobalAttention
)
from torch_geometric.data import Data, Batch


############### LSTM Pooling ###############
class LSTMAttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, batch):
        num_graphs = batch.max().item() + 1
        node_embeddings_list = [x[batch == i] for i in range(num_graphs)]

        pooled_outputs = []
        for node_embeds in node_embeddings_list:
            node_embeds = node_embeds.unsqueeze(0)  # [1, num_nodes, input_dim]
            h_0 = torch.zeros(1, 1, node_embeds.size(2), device=x.device)
            c_0 = torch.zeros(1, 1, node_embeds.size(2), device=x.device)
            lstm_out, _ = self.lstm(node_embeds, (h_0, c_0))

            attention_weights = F.softmax(self.attention(lstm_out.squeeze(0)), dim=0)
            graph_embedding = torch.sum(attention_weights * lstm_out.squeeze(0), dim=0)
            pooled_outputs.append(graph_embedding)

        return torch.stack(pooled_outputs, dim=0)


############### GRU Pooling ###############
class GRUAttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, batch):
        batch_size = batch.max().item() + 1
        max_num_nodes = (batch == 0).sum().item()
        x_packed = torch.zeros((batch_size, max_num_nodes, x.size(1)), device=x.device)

        for i in range(batch_size):
            nodes_in_batch = x[batch == i].unsqueeze(0)
            h, _ = self.gru(nodes_in_batch)
            attention_weights = F.softmax(self.attention(h), dim=1)
            x_packed[i] = torch.sum(attention_weights * h, dim=1)

        return x_packed[:, 0, :]


############### Main Model ###############
class GINGAT(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_channels, out_channels, heads,
                 dropout, pooling_type, num_tasks, use_dummy=True):
        super().__init__()
        self.use_dummy = use_dummy
        self.pooling_type = pooling_type

        # === Graph backbone ===
        self.graph_conv1 = GINEConv(nn.Sequential(
            nn.Linear(node_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ), edge_dim=edge_dim)
        self.graph_bn1 = BatchNorm(hidden_channels)

        self.graph_conv2 = GINEConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ), edge_dim=edge_dim)
        self.graph_bn2 = BatchNorm(hidden_channels)

        self.graph_conv3 = GINEConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ), edge_dim=edge_dim)
        self.graph_bn3 = BatchNorm(hidden_channels)

        self.graph_conv4 = GINEConv(nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        ), edge_dim=edge_dim)
        self.graph_bn4 = BatchNorm(out_channels)

        # === Pooling ===
        if pooling_type == 'lstm':
            self.pooling = LSTMAttentionPooling(out_channels, out_channels)
        elif pooling_type == 'gru':
            self.pooling = GRUAttentionPooling(out_channels, out_channels)
        elif pooling_type == 'attention':
            self.pooling = GlobalAttention(gate_nn=nn.Linear(out_channels, 1))
        elif pooling_type == 'mean':
            self.pooling = global_mean_pool
        elif pooling_type == 'max':
            self.pooling = global_max_pool
        elif pooling_type == 'sum':
            self.pooling = global_add_pool
        else:
            raise ValueError("Pooling must be one of lstm, gru, attention, mean, max, sum")

        # === Dummy graph branch (only if use_dummy) ===
        if self.use_dummy:
            self.node_conv1 = GATv2Conv(out_channels, hidden_channels, heads=heads, concat=False)
            self.node_bn1 = BatchNorm(hidden_channels)
        else:
            self.node_conv1 = None
            self.node_bn1 = None

        # === Projection for ablation (no dummy) ===
        if not self.use_dummy:
            self.proj = nn.Linear(out_channels, hidden_channels)
        else:
            self.proj = None

        # === Output head ===
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_tasks)
        self.dropout = nn.Dropout(dropout)

        self.last_attention = None
        self.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch, data):
        device = x.device
        edge_attr = edge_attr.float().to(device)

        # === GNN encoder ===
        x = self.graph_conv1(x, edge_index, edge_attr)
        x = self.graph_bn1(x); x = F.relu(x); x = self.dropout(x)
        x = self.graph_conv2(x, edge_index, edge_attr)
        x = self.graph_bn2(x); x = F.relu(x); x = self.dropout(x)
        x = self.graph_conv3(x, edge_index, edge_attr)
        x = self.graph_bn3(x); x = F.relu(x); x = self.dropout(x)
        x = self.graph_conv4(x, edge_index, edge_attr)
        x = self.graph_bn4(x); x = F.relu(x)

        graph_out = self.pooling(x, batch)

        # raw features
        fingerprints = [data.ECFP.to(device), data.Topological.to(device),
                        data.MACCS.to(device), data.EState.to(device)]
        descriptors = [data.Rdkit2D.to(device), data.Phar2D.to(device)]
        features = fingerprints + descriptors
        features = [f.unsqueeze(0) if f.dim() == 1 else f for f in features]

        if self.use_dummy:
            # === Dummy graph branch ===
            dummy_graphs = []
            for i in range(graph_out.size(0)):
                dummy_graph = self.create_dummy_graph(
                    graph_out[i].unsqueeze(0),
                    [f[i].unsqueeze(0) for f in features],
                    device
                )
                dummy_graphs.append(dummy_graph)

            batched_dummy = Batch.from_data_list(dummy_graphs).to(device)
            x_dummy, edge_index_dummy = batched_dummy.x, batched_dummy.edge_index

            out1 = self.node_conv1(x_dummy, edge_index_dummy, return_attention_weights=True)
            if isinstance(out1, tuple):
                x_node, (attn_edge_index, attn_alpha) = out1
            else:
                x_node, attn_edge_index, attn_alpha = out1, None, None

            x_node = self.node_bn1(x_node); x_node = F.relu(x_node)

            self.last_attention = {
                "edge_index": attn_edge_index.detach().cpu() if attn_edge_index is not None else None,
                "alpha": attn_alpha.detach().cpu() if attn_alpha is not None else None
            }

            # central nodes
            num_feats = len(features)
            stride = num_feats + 1
            central_indices = torch.arange(0, len(dummy_graphs) * stride, stride, device=device)
            x_central = x_node[central_indices]

            x_final = F.relu(self.fc1(x_central))
            x_final = self.dropout(x_final)
            return self.fc2(x_final)

        else:
            # === Ablation: concat + projection ===
            feat_cat = torch.cat([graph_out] + features, dim=1)
            feat_proj = F.relu(self.proj(feat_cat))

            x_final = F.relu(self.fc1(feat_proj))
            x_final = self.dropout(x_final)
            self.last_attention = None
            return self.fc2(x_final)

    def create_dummy_graph(self, graph_embedding, features, device):
        central_node = graph_embedding.squeeze(0).unsqueeze(0)
        feat_nodes = [f.view(1, -1) if f.dim() > 2 else f for f in features]
        node_features = torch.cat([central_node] + feat_nodes, dim=0)

        edges = [[0, i] for i in range(1, len(feat_nodes) + 1)]
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

        return Data(x=node_features, edge_index=edge_index)

    def reset_parameters(self):
        self.graph_conv1.reset_parameters(); self.graph_bn1.reset_parameters()
        self.graph_conv2.reset_parameters(); self.graph_bn2.reset_parameters()
        self.graph_conv3.reset_parameters(); self.graph_bn3.reset_parameters()
        self.graph_conv4.reset_parameters(); self.graph_bn4.reset_parameters()

        if self.pooling_type in ['lstm', 'gru']:
            try:
                self.pooling.lstm.reset_parameters()
                self.pooling.attention.reset_parameters()
            except Exception:
                pass

        if self.use_dummy:
            self.node_conv1.reset_parameters()
            self.node_bn1.reset_parameters()

        self.fc1.reset_parameters(); self.fc2.reset_parameters()
        if not self.use_dummy and self.proj is not None:
            self.proj.reset_parameters()
