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
        pooled_outputs = []
        for i in range(num_graphs):
            node_embeds = x[batch == i].unsqueeze(0)  # [1, num_nodes_in_graph, input_dim]
            h_0 = torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size, device=x.device)
            c_0 = torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size, device=x.device)
            lstm_out, _ = self.lstm(node_embeds, (h_0, c_0)) # [1, num_nodes_in_graph, hidden_dim]
            attention_weights = F.softmax(self.attention(lstm_out.squeeze(0)), dim=0) # [num_nodes_in_graph, 1]
            graph_embedding = torch.sum(attention_weights * lstm_out.squeeze(0), dim=0) # [hidden_dim]
            pooled_outputs.append(graph_embedding)
        return torch.stack(pooled_outputs, dim=0) # [num_graphs, hidden_dim]


############### GRU Pooling ###############
class GRUAttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, batch):
        pooled_outputs = []
        num_graphs = batch.max().item() + 1
        for i in range(num_graphs):
            nodes_in_graph = x[batch == i].unsqueeze(0) # [1, num_nodes_in_graph, input_dim]
            h_0 = torch.zeros(self.gru.num_layers, 1, self.gru.hidden_size, device=x.device)
            gru_out, _ = self.gru(nodes_in_graph, h_0) # [1, num_nodes_in_graph, hidden_dim]
            attention_weights = F.softmax(self.attention(gru_out.squeeze(0)), dim=0) # [num_nodes_in_graph, 1]
            graph_embedding = torch.sum(attention_weights * gru_out.squeeze(0), dim=0) # [hidden_dim]
            pooled_outputs.append(graph_embedding)
        return torch.stack(pooled_outputs, dim=0) # [num_graphs, hidden_dim]


############### Main Model (GINGAT) ###############
class GINGAT(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_channels, out_channels, heads,
                 dropout, pooling_type, num_tasks, use_dummy=True, feature_mode="both"):
        """
        Args:
            feature_mode (str): Controls which handcrafted features to use.
                Options: "fingerprints", "descriptors", "both".
        """
        super().__init__()
        self.use_dummy = use_dummy
        self.pooling_type = pooling_type
        self.feature_mode = feature_mode 

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # === Graph backbone (GINEConv layers) ===
        self.graph_conv1 = GINEConv(nn.Sequential(
            nn.Linear(node_dim, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ), edge_dim=edge_dim)
        self.graph_bn1 = BatchNorm(hidden_channels)

        self.graph_conv2 = GINEConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ), edge_dim=edge_dim)
        self.graph_bn2 = BatchNorm(hidden_channels)

        self.graph_conv3 = GINEConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ), edge_dim=edge_dim)
        self.graph_bn3 = BatchNorm(hidden_channels)

        self.graph_conv4 = GINEConv(nn.Sequential(
            nn.Linear(hidden_channels, out_channels), nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        ), edge_dim=edge_dim)
        self.graph_bn4 = BatchNorm(out_channels)

        # === Graph Pooling Layer ===
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
            raise ValueError("Pooling must be one of 'lstm', 'gru', 'attention', 'mean', 'max', 'sum'")

        # === Dummy graph branch ===
        if self.use_dummy:
            self.node_conv1 = GATv2Conv(out_channels, hidden_channels, heads=heads, concat=False)
            self.node_bn1 = BatchNorm(hidden_channels)
            self.residual_proj = nn.Linear(out_channels, hidden_channels)
        else:
            self.node_conv1 = None
            self.node_bn1 = None
            self.residual_proj = None
            self.ablation_proj = None  # Will be initialized dynamically

        # === Output head ===
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_tasks)
        self.dropout = nn.Dropout(dropout)

        self.last_attention = None
        self.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch, data):
        device = x.device
        edge_attr = edge_attr.float().to(device)

        # === GNN Encoder ===
        x = self.graph_conv1(x, edge_index, edge_attr)
        x = self.graph_bn1(x); x = F.relu(x); x = self.dropout(x)
        x = self.graph_conv2(x, edge_index, edge_attr)
        x = self.graph_bn2(x); x = F.relu(x); x = self.dropout(x)
        x = self.graph_conv3(x, edge_index, edge_attr)
        x = self.graph_bn3(x); x = F.relu(x); x = self.dropout(x)
        x = self.graph_conv4(x, edge_index, edge_attr)
        x = self.graph_bn4(x); x = F.relu(x)  # [num_nodes_in_batch, out_channels]

        # === Graph Pooling ===
        graph_out = self.pooling(x, batch)  # [batch_size, out_channels]

        # === Feature Selection based on `feature_mode` ===
        if self.feature_mode == "fingerprints":
            selected_features = [
                data.ECFP.to(device),
                data.Topological.to(device),
                data.MACCS.to(device),
                data.EState.to(device)
            ]
        elif self.feature_mode == "descriptors":
            selected_features = [
                data.Rdkit2D.to(device),
                data.Phar2D.to(device)
            ]
        elif self.feature_mode == "both":
            selected_features = [
                data.ECFP.to(device),
                data.Topological.to(device),
                data.MACCS.to(device),
                data.EState.to(device),
                data.Rdkit2D.to(device),
                data.Phar2D.to(device)
            ]
        else:
            raise ValueError(f"Invalid feature_mode: {self.feature_mode}. Choose from 'fingerprints', 'descriptors', 'both'.")

        # Ensure all features are 2D: [batch_size, feature_dim]
        features_2d = []
        for f in selected_features:
            if f.dim() == 1:
                features_2d.append(f.unsqueeze(1))  # [batch_size, 1]
            else:
                features_2d.append(f.view(graph_out.size(0), -1))  # [batch_size, N]

        # === Apply Layer Normalization ===
        graph_out = F.layer_norm(graph_out, graph_out.size()[1:])
        normalized_features = [F.layer_norm(f, f.size()[1:]) for f in features_2d]

        if self.use_dummy:
            # === Process via Dummy Graph & GAT ===
            dummy_graphs = []
            for i in range(graph_out.size(0)):
                dummy_graph = self.create_dummy_graph(
                    graph_out[i].unsqueeze(0),
                    [f[i].unsqueeze(0) for f in normalized_features],
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

            # Extract central (target) node embeddings
            num_feats_per_graph = len(normalized_features)
            stride = num_feats_per_graph + 1
            central_indices = torch.arange(0, len(dummy_graphs) * stride, stride, device=device)
            processed_central = x_node[central_indices]  # [batch_size, hidden_channels]

            # Add Residual Connection
            projected_central = self.residual_proj(graph_out)  # [batch_size, hidden_channels]
            x_processed = processed_central + projected_central  # [batch_size, hidden_channels]

        else:
            # === Ablation: Direct Concatenation ===
            feat_cat = torch.cat([graph_out] + normalized_features, dim=1)  # [batch_size, total_dim]

            if self.ablation_proj is None:
                total_concat_dim = feat_cat.size(1)
                self.ablation_proj = nn.Linear(total_concat_dim, self.hidden_channels).to(device)

            x_processed = F.relu(self.ablation_proj(feat_cat))  # [batch_size, hidden_channels]
            self.last_attention = None

        # === Final Prediction Head ===
        x_final = F.relu(self.fc1(x_processed))
        x_final = self.dropout(x_final)
        return self.fc2(x_final)

    def create_dummy_graph(self, graph_embedding, features, device):
        """
        Creates a dummy graph Data object.
        Args:
            graph_embedding: Tensor of shape [1, out_channels] for the central node.
            features: List of tensors, each of shape [1, feature_dim] for peripheral nodes.
        Returns:
            PyG Data object.
        """
        node_features = torch.cat([graph_embedding] + features, dim=0)  # [1 + N, D]
        edges = [[0, i] for i in range(1, len(features) + 1)]  # Central node (0) -> all features
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        return Data(x=node_features, edge_index=edge_index)

    def reset_parameters(self):
        # Reset GNN backbone
        for conv, bn in [
            (self.graph_conv1, self.graph_bn1),
            (self.graph_conv2, self.graph_bn2),
            (self.graph_conv3, self.graph_bn3),
            (self.graph_conv4, self.graph_bn4)
        ]:
            conv.reset_parameters()
            bn.reset_parameters()

        # Reset pooling
        if hasattr(self.pooling, 'reset_parameters'):
            self.pooling.reset_parameters()
        elif self.pooling_type == 'lstm':
            self.pooling.lstm.reset_parameters()
            self.pooling.attention.reset_parameters()
        elif self.pooling_type == 'gru':
            self.pooling.gru.reset_parameters()
            self.pooling.attention.reset_parameters()

        # Reset dummy graph components
        if self.use_dummy:
            if self.node_conv1 is not None:
                self.node_conv1.reset_parameters()
            if self.node_bn1 is not None:
                self.node_bn1.reset_parameters()
            if self.residual_proj is not None:
                self.residual_proj.reset_parameters()
        else:
            if self.ablation_proj is not None:
                self.ablation_proj.reset_parameters()

        # Reset final layers
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()