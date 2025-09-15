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
    # Kept original __init__ signature (num_layers=1 default)
    def __init__(self, input_dim, hidden_dim, num_layers=1): 
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, batch):
        num_graphs = batch.max().item() + 1
        pooled_outputs = []
        for i in range(num_graphs):
            node_embeds = x[batch == i].unsqueeze(0)  # [1, num_nodes_in_graph, input_dim]
            
            # Initialize hidden and cell states for this graph
            # States should match the LSTM's num_layers and hidden_size
            h_0 = torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size, device=x.device)
            c_0 = torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size, device=x.device)
            
            lstm_out, _ = self.lstm(node_embeds, (h_0, c_0)) # lstm_out: [1, num_nodes_in_graph, hidden_dim]

            attention_weights = F.softmax(self.attention(lstm_out.squeeze(0)), dim=0) # [num_nodes_in_graph, 1]
            graph_embedding = torch.sum(attention_weights * lstm_out.squeeze(0), dim=0) # [hidden_dim]
            pooled_outputs.append(graph_embedding)

        return torch.stack(pooled_outputs, dim=0) # [num_graphs, hidden_dim]


############### GRU Pooling ###############
class GRUAttentionPooling(nn.Module):
    # Kept original __init__ signature (num_layers=1 default implicitly)
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # GRU defaults to 1 layer if num_layers is not specified
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True) 
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, batch):
        pooled_outputs = []
        num_graphs = batch.max().item() + 1
        
        for i in range(num_graphs):
            nodes_in_graph = x[batch == i].unsqueeze(0) # [1, num_nodes_in_graph, input_dim]
            
            # Initialize hidden state for this graph
            h_0 = torch.zeros(self.gru.num_layers, 1, self.gru.hidden_size, device=x.device) # Corrected num_layers
            
            gru_out, _ = self.gru(nodes_in_graph, h_0) # gru_out: [1, num_nodes_in_graph, hidden_dim]
            
            attention_weights = F.softmax(self.attention(gru_out.squeeze(0)), dim=0) # [num_nodes_in_graph, 1]
            graph_embedding = torch.sum(attention_weights * gru_out.squeeze(0), dim=0) # [hidden_dim]
            pooled_outputs.append(graph_embedding)
            
        return torch.stack(pooled_outputs, dim=0) # [num_graphs, hidden_dim]


############### Main Model ###############
class GINGAT(nn.Module):
    # Original __init__ signature maintained
    def __init__(self, node_dim, edge_dim, hidden_channels, out_channels, heads,
                 dropout, pooling_type, num_tasks, use_dummy=True):
        super().__init__()
        self.use_dummy = use_dummy
        self.pooling_type = pooling_type
        
        # Store for later use and dimension consistency
        self.out_channels = out_channels 
        self.hidden_channels = hidden_channels 

        # === Graph backbone (GINEConv layers) ===
        # Layer 1
        self.graph_conv1 = GINEConv(nn.Sequential(
            nn.Linear(node_dim, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ), edge_dim=edge_dim)
        self.graph_bn1 = BatchNorm(hidden_channels)

        # Layer 2
        self.graph_conv2 = GINEConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ), edge_dim=edge_dim)
        self.graph_bn2 = BatchNorm(hidden_channels)

        # Layer 3
        self.graph_conv3 = GINEConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ), edge_dim=edge_dim)
        self.graph_bn3 = BatchNorm(hidden_channels)

        # Layer 4 (Output layer of GNN backbone)
        self.graph_conv4 = GINEConv(nn.Sequential(
            nn.Linear(hidden_channels, out_channels), nn.ReLU(), # Output here is `out_channels`
            nn.Linear(out_channels, out_channels)
        ), edge_dim=edge_dim)
        self.graph_bn4 = BatchNorm(out_channels) # BatchNorm for `out_channels`

        # === Graph Pooling Layer ===
        # Pooling layer takes `out_channels` as input and should output `out_channels` for consistency
        # For LSTMAttentionPooling and GRUAttentionPooling, hidden_dim is set to out_channels
        if pooling_type == 'lstm':
            # num_layers defaults to 1 in LSTMAttentionPooling's __init__
            self.pooling = LSTMAttentionPooling(out_channels, out_channels)
        elif pooling_type == 'gru':
            # num_layers defaults to 1 implicitly in GRUAttentionPooling's __init__
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

        # === Dummy graph branch (GATv2Conv layer) ===
        if self.use_dummy:
            # GATv2Conv takes `out_channels` (from graph_out) as input and outputs `hidden_channels`
            self.node_conv1 = GATv2Conv(out_channels, hidden_channels, heads=heads, concat=False)
            self.node_bn1 = BatchNorm(hidden_channels)
        else:
            self.node_conv1 = None
            self.node_bn1 = None
            # Ablation path: Linear projection created dynamically in `forward`
            self.ablation_proj = None # Placeholder for dynamic initialization

        # === Output head (Fully Connected Layers) ===
        # Input to fc1 is `hidden_channels` from either:
        # 1. `node_conv1` output (dummy path)
        # 2. `ablation_proj` output (ablation path)
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_tasks)
        self.dropout = nn.Dropout(dropout)

        self.last_attention = None # To store attention weights from GATv2Conv if needed for analysis
        self.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch, data):
        device = x.device
        edge_attr = edge_attr.float().to(device)

        # === GNN encoder (Message Passing) ===
        x = self.graph_conv1(x, edge_index, edge_attr)
        x = self.graph_bn1(x); x = F.relu(x); x = self.dropout(x)
        x = self.graph_conv2(x, edge_index, edge_attr)
        x = self.graph_bn2(x); x = F.relu(x); x = self.dropout(x)
        x = self.graph_conv3(x, edge_index, edge_attr)
        x = self.graph_bn3(x); x = F.relu(x); x = self.dropout(x)
        x = self.graph_conv4(x, edge_index, edge_attr)
        x = self.graph_bn4(x); x = F.relu(x) # x is [num_nodes_in_batch, out_channels]

        # === Graph Pooling ===
        graph_out = self.pooling(x, batch) # graph_out is [batch_size, out_channels]

        # Get raw features from the `data` object
        # Ensure features are correctly shaped [batch_size, feature_dim]
        # `data.attribute` from PyG's Batch object is already batched
        fingerprints = [data.ECFP.to(device), data.Topological.to(device),
                        data.MACCS.to(device), data.EState.to(device)]
        descriptors = [data.Rdkit2D.to(device), data.Phar2D.to(device)]
        
        # Flatten features if they are 1D (e.g., [batch_size]), or ensure they are [batch_size, N]
        features_2d = []
        for f in (fingerprints + descriptors):
            if f.dim() == 1: # If it's just [batch_size] (e.g., a single value per graph)
                features_2d.append(f.unsqueeze(1)) # Make it [batch_size, 1]
            else: # Otherwise, it's already [batch_size, feature_dim] or higher
                features_2d.append(f.view(graph_out.size(0), -1)) # Reshape to [batch_size, some_dim]

        if self.use_dummy:
            # === Dummy graph branch processing ===
            dummy_graphs = []
            for i in range(graph_out.size(0)): # Iterate through each graph in the batch
                # create_dummy_graph expects:
                # graph_embedding: [1, out_channels]
                # features: list of [1, feature_dim] tensors
                dummy_graph = self.create_dummy_graph(
                    graph_out[i].unsqueeze(0),       # Isolate current graph_embedding [1, out_channels]
                    [f[i].unsqueeze(0) for f in features_2d], # Isolate current features [1, feature_dim]
                    device
                )
                dummy_graphs.append(dummy_graph)

            # Batch the dummy graphs for GATv2Conv
            batched_dummy = Batch.from_data_list(dummy_graphs).to(device)
            x_dummy, edge_index_dummy = batched_dummy.x, batched_dummy.edge_index # x_dummy is [total_nodes_in_dummy_batch, out_channels]

            # GATv2Conv forward pass
            out1 = self.node_conv1(x_dummy, edge_index_dummy, return_attention_weights=True)
            if isinstance(out1, tuple):
                x_node, (attn_edge_index, attn_alpha) = out1 # x_node is [total_nodes_in_dummy_batch, hidden_channels]
            else:
                x_node, attn_edge_index, attn_alpha = out1, None, None

            x_node = self.node_bn1(x_node); x_node = F.relu(x_node)

            # Store attention weights if needed for analysis/visualization
            self.last_attention = {
                "edge_index": attn_edge_index.detach().cpu() if attn_edge_index is not None else None,
                "alpha": attn_alpha.detach().cpu() if attn_alpha is not None else None
            }

            # Extract the central nodes' features from the GATv2Conv output
            num_feats_per_graph = len(features_2d) 
            stride = num_feats_per_graph + 1 # Each dummy graph has 1 central + num_feats peripheral
            central_indices = torch.arange(0, len(dummy_graphs) * stride, stride, device=device)
            
            x_processed = x_node[central_indices] # [batch_size, hidden_channels]
            
        else: # Ablation path (use_dummy=False)
            # === Concatenate graph_out and features, then project ===
            # `graph_out` is [batch_size, out_channels]
            # `features_2d` is a list of [batch_size, feature_dim_i]
            feat_cat = torch.cat([graph_out] + features_2d, dim=1) # [batch_size, total_concat_dim]
            
            # Dynamically create ablation_proj if it's the first forward pass
            if self.ablation_proj is None:
                total_concat_dim = feat_cat.size(1)
                # Project concatenated features back to `hidden_channels` to align with the dummy branch output
                self.ablation_proj = nn.Linear(total_concat_dim, self.hidden_channels).to(device)
                # print(f"Initialized ablation_proj: Input dim {total_concat_dim}, Output dim {self.hidden_channels}")

            x_processed = F.relu(self.ablation_proj(feat_cat)) # [batch_size, hidden_channels]
            self.last_attention = None # No attention mechanism in this path

        # === Final output head (FC Layers) ===
        # `x_processed` is consistently [batch_size, hidden_channels]
        x_final = F.relu(self.fc1(x_processed))
        x_final = self.dropout(x_final)
        return self.fc2(x_final)

    def create_dummy_graph(self, graph_embedding, features, device):
        # `graph_embedding`: [1, out_channels] (central node)
        # `features`: list of [1, feature_dim] tensors (peripheral nodes)

        # Concatenate all node features for the dummy graph
        node_features = torch.cat([graph_embedding] + features, dim=0) # [1 + num_feats, max_feature_dim]
        
        # Create edges from the central node (index 0) to all peripheral feature nodes
        edges = [[0, i] for i in range(1, len(features) + 1)]
        
        # Transpose to get [2, num_edges] format
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

        return Data(x=node_features, edge_index=edge_index)

    def reset_parameters(self):
        # Reset GNN backbone layers
        self.graph_conv1.reset_parameters(); self.graph_bn1.reset_parameters()
        self.graph_conv2.reset_parameters(); self.graph_bn2.reset_parameters()
        self.graph_conv3.reset_parameters(); self.graph_bn3.reset_parameters()
        self.graph_conv4.reset_parameters(); self.graph_bn4.reset_parameters()

        # Reset pooling layers
        if hasattr(self.pooling, 'reset_parameters'): # For global_mean_pool etc.
            self.pooling.reset_parameters()
        elif self.pooling_type == 'lstm': # Specific reset for custom LSTM pooling
            self.pooling.lstm.reset_parameters()
            self.pooling.attention.reset_parameters()
        elif self.pooling_type == 'gru': # Specific reset for custom GRU pooling
            self.pooling.gru.reset_parameters()
            self.pooling.attention.reset_parameters()

        # Reset dummy graph specific layers
        if self.use_dummy:
            if self.node_conv1 is not None: self.node_conv1.reset_parameters()
            if self.node_bn1 is not None: self.node_bn1.reset_parameters()
        else: # Reset ablation projection layer
            if self.ablation_proj is not None: self.ablation_proj.reset_parameters()

        # Reset final FC layers
        self.fc1.reset_parameters(); self.fc2.reset_parameters()