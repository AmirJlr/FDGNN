import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv, GINEConv, BatchNorm,  
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
                 dropout, pooling_type, num_tasks, use_dummy=True, feature_mode="both",
                 num_gin_layers=4, num_gat_layers=1):  
        """
        Args:
            feature_mode (str): Controls which handcrafted features to use.
                Options: "fps", "descs", "both".
            num_gin_layers (int): Number of GINEConv layers in the GNN backbone.
            num_gat_layers (int): Number of GAT layers to apply on the dummy graph.
        """
        super().__init__()
        self.use_dummy = use_dummy
        self.pooling_type = pooling_type
        self.feature_mode = feature_mode
        self.num_gin_layers = num_gin_layers  # <-- Store number of GIN layers
        self.num_gat_layers = num_gat_layers

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # === Graph backbone (Stack of GINEConv layers) ===
        self.graph_convs = nn.ModuleList()
        self.graph_bns = nn.ModuleList()

        for i in range(self.num_gin_layers):
            in_dim = node_dim if i == 0 else hidden_channels
            out_dim = hidden_channels if i < self.num_gin_layers - 1 else out_channels

            self.graph_convs.append(
                GINEConv(nn.Sequential(
                    nn.Linear(in_dim, out_dim), nn.ReLU(),
                    nn.Linear(out_dim, out_dim)
                ), edge_dim=edge_dim)
            )
            self.graph_bns.append(BatchNorm(out_dim))

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
            # Stack of GAT layers
            self.node_convs = nn.ModuleList()
            self.node_bns = nn.ModuleList()

            for i in range(self.num_gat_layers):
                in_dim = out_channels if i == 0 else hidden_channels
                out_dim = hidden_channels
                # Use GATConv (original) for more sensitive attention
                self.node_convs.append(GATConv(in_dim, out_dim, heads=heads, concat=False))
                self.node_bns.append(BatchNorm(out_dim))

            # Residual projection for the first layer only (if dimensions change)
            if out_channels != hidden_channels:
                self.residual_proj = nn.Linear(out_channels, hidden_channels)
            else:
                self.residual_proj = None
        else:
            self.node_convs = None
            self.node_bns = None
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

        # === GNN Encoder (Stack of GINEConv layers) ===
        for i, (conv, bn) in enumerate(zip(self.graph_convs, self.graph_bns)):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            if i < self.num_gin_layers - 1:  # Apply dropout except on last layer
                x = self.dropout(x)

        # === Graph Pooling ===
        graph_out = self.pooling(x, batch)  # [batch_size, out_channels]

        # === Feature Selection based on `feature_mode` ===
        if self.feature_mode == "fps":
            selected_features = [
                data.ECFP.to(device),
                data.Topological.to(device),
                data.MACCS.to(device),
                data.EState.to(device)
            ]
        elif self.feature_mode == "descs":
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
            # === Process via Complete Dummy Graph & GAT Stack ===
            dummy_graphs = []
            for i in range(graph_out.size(0)):
                dummy_graph = self.create_complete_dummy_graph(
                    graph_out[i].unsqueeze(0),
                    [f[i].unsqueeze(0) for f in normalized_features],
                    device
                )
                dummy_graphs.append(dummy_graph)

            batched_dummy = Batch.from_data_list(dummy_graphs).to(device)
            x_dummy, edge_index_dummy = batched_dummy.x, batched_dummy.edge_index

            # === CRITICAL: Store the batch vector for attention visualization ===
            self.last_attention["batch"] = batched_dummy.batch 

            # Apply stack of GAT layers
            for i, (conv, bn) in enumerate(zip(self.node_convs, self.node_bns)):
                if i == 0 and self.residual_proj is not None:
                    # Save for residual after first layer
                    initial_x = x_dummy

                out = conv(x_dummy, edge_index_dummy, return_attention_weights=True)
                if isinstance(out, tuple):
                    x_dummy, (attn_edge_index, attn_alpha) = out
                else:
                    x_dummy, attn_edge_index, attn_alpha = out, None, None

                x_dummy = bn(x_dummy)
                x_dummy = F.relu(x_dummy)

                # Apply residual connection after first layer if dimensions changed
                if i == 0 and self.residual_proj is not None:
                    x_dummy = x_dummy + self.residual_proj(initial_x)

            self.last_attention = {
                "edge_index": attn_edge_index.detach().cpu() if attn_edge_index is not None else None,
                "alpha": attn_alpha.detach().cpu() if attn_alpha is not None else None
            }

            # Extract central (target) node embeddings (Node 0 in each dummy graph)
            num_feats_per_graph = len(normalized_features)
            stride = num_feats_per_graph + 1
            central_indices = torch.arange(0, len(dummy_graphs) * stride, stride, device=device)
            x_processed = x_dummy[central_indices]  # [batch_size, hidden_channels]

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

    def create_complete_dummy_graph(self, graph_embedding, features, device):
        """
        Creates a COMPLETE dummy graph Data object.
        The central node (index 0) is ALWAYS the graph_embedding.
        Args:
            graph_embedding: Tensor of shape [1, out_channels] for the central node.
            features: List of tensors, each of shape [1, feature_dim] for peripheral nodes.
        Returns:
            PyG Data object with a complete graph structure.
        """
        node_features = torch.cat([graph_embedding] + features, dim=0)  # [1 + N, D]
        num_nodes = node_features.size(0)

        # Create a complete graph: connect every node to every other node (including self-loops)
        edge_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):  # This includes self-loop (i, i)
                edge_list.append([i, j])

        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
        return Data(x=node_features, edge_index=edge_index)

    def reset_parameters(self):
        # Reset GNN backbone
        for conv, bn in zip(self.graph_convs, self.graph_bns):
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
            for conv, bn in zip(self.node_convs, self.node_bns):
                conv.reset_parameters()
                bn.reset_parameters()
            if self.residual_proj is not None:
                self.residual_proj.reset_parameters()
        else:
            if self.ablation_proj is not None:
                self.ablation_proj.reset_parameters()

        # Reset final layers
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()