##### GAT-LSTM-GIN
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINConv, BatchNorm, GINEConv
from torch_geometric.data import Data, Batch

from torch_geometric.nn import (
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    GlobalAttention
)



############### LSTM Pooling ###############
class LSTMAttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMAttentionPooling, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, batch):
        # x: [num_nodes, input_dim]
        # batch: [num_nodes] indicates which graph each node belongs to

        # Group nodes by graph
        num_graphs = batch.max().item() + 1
        node_embeddings_list = [x[batch == i] for i in range(num_graphs)]

        pooled_outputs = []
        for node_embeds in node_embeddings_list:
            node_embeds = node_embeds.unsqueeze(0)  # [1, num_nodes, input_dim]
            h_0 = torch.zeros(1, 1, node_embeds.size(2)).to(x.device)
            c_0 = torch.zeros(1, 1, node_embeds.size(2)).to(x.device)
            lstm_out, _ = self.lstm(node_embeds, (h_0, c_0))  # [1, num_nodes, hidden_dim]

            attention_weights = F.softmax(self.attention(lstm_out.squeeze(0)), dim=0)  # [num_nodes, 1]
            graph_embedding = torch.sum(attention_weights * lstm_out.squeeze(0), dim=0)  # [hidden_dim]
            pooled_outputs.append(graph_embedding)

        graph_embeddings = torch.stack(pooled_outputs, dim=0)
        return graph_embeddings

############### GRU Pooling ###############
class GRUAttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUAttentionPooling, self).__init__()
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


class GINGAT(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_channels, out_channels, heads,
                 dropout, pooling_type, num_tasks, use_dummy=True):
        super(GINGAT, self).__init__()
        self.use_dummy = use_dummy
        self.pooling_type = pooling_type

        # Graph-level GNN (unchanged)
        self.graph_conv1 = GINEConv(nn.Sequential(nn.Linear(node_dim, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)), edge_dim=edge_dim)
        self.graph_bn1 = BatchNorm(hidden_channels)

        self.graph_conv2 = GINEConv(nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)), edge_dim=edge_dim)
        self.graph_bn2 = BatchNorm(hidden_channels)

        self.graph_conv3 = GINEConv(nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)), edge_dim=edge_dim)
        self.graph_bn3 = BatchNorm(hidden_channels)
        
        self.graph_conv4 = GINEConv(nn.Sequential(nn.Linear(hidden_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels)), edge_dim=edge_dim)
        self.graph_bn4 = BatchNorm(out_channels)

        # Pooling selection
        if pooling_type == 'lstm':
            self.pooling = LSTMAttentionPooling(out_channels, out_channels)
        elif pooling_type == 'gru':
            self.pooling = GRUAttentionPooling(out_channels, out_channels)
        elif pooling_type == 'attention':
            from torch_geometric.nn import GlobalAttention
            self.pooling = GlobalAttention(gate_nn=nn.Linear(out_channels, 1))
        elif pooling_type == 'mean':
            from torch_geometric.nn import global_mean_pool
            self.pooling = global_mean_pool
        elif pooling_type == 'max':
            from torch_geometric.nn import global_max_pool
            self.pooling = global_max_pool
        elif pooling_type == 'sum':
            from torch_geometric.nn import global_add_pool
            self.pooling = global_add_pool
        else:
            raise ValueError("please use lstm, gru, attention, mean, max or sum for pooling")

        # Node-level GAT: request attention weights when forward called
        self.node_conv1 = GATv2Conv(out_channels, hidden_channels, heads=heads, concat=False)
        self.node_bn1 = BatchNorm(hidden_channels)

        # Output MLP
        # If use_dummy==False we'll operate on concatenated features; choose sizes accordingly
        # We'll compute the input dim dynamically in forward; keep fc1/out as flexible.
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_tasks)
        self.dropout = nn.Dropout(dropout)

        # storage for attention weights (populated in forward)
        self.last_attention = None  # will be a dict {'edge_index': Tensor, 'alpha': Tensor}
        self.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch, data):
        device = x.device
        edge_attr = edge_attr.float().to(device)

        # Graph-level encoder
        x = self.graph_conv1(x, edge_index, edge_attr)
        x = self.graph_bn1(x); x = F.relu(x); x = self.dropout(x)

        x = self.graph_conv2(x, edge_index, edge_attr)
        x = self.graph_bn2(x); x = F.relu(x); x = self.dropout(x)

        x = self.graph_conv3(x, edge_index, edge_attr)
        x = self.graph_bn3(x); x = F.relu(x); x = self.dropout(x)

        x = self.graph_conv4(x, edge_index, edge_attr)
        x = self.graph_bn4(x); x = F.relu(x)

        # Pooling -> graph-level embeddings (batch_size, out_channels)
        graph_out = self.pooling(x, batch)

        # Extract fingerprints and descriptors tensors from `data`
        fingerprints = [data.ECFP.to(device), data.Topological.to(device), data.MACCS.to(device), data.EState.to(device)]
        descriptors = [data.Rdkit2D.to(device), data.Phar2D.to(device)]
        features = fingerprints + descriptors
        features = [f.unsqueeze(0) if f.dim() == 1 else f for f in features]  # make sure shape (batch, feat_dim)

        if self.use_dummy:
            # --- build dummy-graph per sample and run node-level GAT over the batched dummy graphs ---
            dummy_graphs = []
            for i in range(graph_out.size(0)):
                dummy_graph = self.create_dummy_graph(graph_out[i].unsqueeze(0), [f[i].unsqueeze(0) for f in features], device)
                dummy_graphs.append(dummy_graph)
            batched_dummy_graph = Batch.from_data_list(dummy_graphs).to(device)
            x_dummy, edge_index_dummy = batched_dummy_graph.x, batched_dummy_graph.edge_index

            # Get attention weights from GAT: use return_attention_weights if available in your PyG version
            # Many PyG convs return a tuple (out, (edge_index, alpha)) when return_attention_weights=True
            try:
                out1 = self.node_conv1(x_dummy, edge_index_dummy, return_attention_weights=True)
                # out1 could be (out, (edge_index, alpha))
                if isinstance(out1, tuple) and len(out1) == 2:
                    x_node, attn_tuple = out1
                    attn_edge_index, attn_alpha = attn_tuple
                else:
                    # fallback if API different
                    x_node = out1
                    attn_edge_index = edge_index_dummy
                    attn_alpha = None
            except TypeError:
                # older/newer versions may not accept return_attention_weights param
                x_node = self.node_conv1(x_dummy, edge_index_dummy)
                attn_edge_index = edge_index_dummy
                attn_alpha = None

            x_node = self.node_bn1(x_node); x_node = F.relu(x_node)

            # store attention for later analysis (convert to CPU)
            self.last_attention = {
                'edge_index': attn_edge_index.detach().cpu() if attn_edge_index is not None else None,
                'alpha': attn_alpha.detach().cpu() if attn_alpha is not None else None
            }

            # central nodes are at indices 0, (k+1), 2*(k+1), ... if each dummy has (1 + num_features) nodes
            num_feats = len(features)
            stride = num_feats + 1
            central_node_indices = torch.arange(0, len(dummy_graphs) * stride, stride, device=device)
            x_central = x_node[central_node_indices]

            x_final = F.relu(self.fc1(x_central))
            x_final = self.dropout(x_final)
            predictions = self.fc2(x_final)
            return predictions

        else:
            # --- NO DUMMY GRAPH ablation branch ---
            # Strategy: for each sample, concatenate graph_out[i] with flattened features for that sample,
            # then pass through a small MLP to produce predictions.
            # Ensure features are 2D: (batch, feat_dim); graph_out is (batch, out_channels)
            feat_list = [f for f in features]  # each is (batch, dim)
            feat_cat = torch.cat([graph_out] + feat_list, dim=1)  # (batch, out_channels + sum(feat_dims))

            # Optionally project concatenated features down to hidden size first
            proj = nn.Linear(feat_cat.size(1), self.fc1.in_features).to(device)  # temporary projector
            feat_proj = F.relu(proj(feat_cat))
            x_final = F.relu(self.fc1(feat_proj))
            x_final = self.dropout(x_final)
            predictions = self.fc2(x_final)

            # clear last_attention because we didn't run node-level GAT
            self.last_attention = None
            return predictions

    def create_dummy_graph(self, graph_embedding, features, device):
        # graph_embedding: (1, out_dim)
        # features: list of (1, feat_dim)
        
        # Central node: always 2D
        central_node_features = graph_embedding.squeeze(0).unsqueeze(0)  # (1, out_dim)
        
        # Ensure each feature is also 2D row (1, feat_dim)
        feat_nodes = []
        for f in features:
            f2 = f
            if f2.dim() == 1:  # shape (feat_dim,)
                f2 = f2.unsqueeze(0)  # -> (1, feat_dim)
            elif f2.dim() > 2:
                f2 = f2.view(1, -1)  # flatten any higher dims
            feat_nodes.append(f2)
        
        # Concatenate into node matrix
        node_features = torch.cat([central_node_features] + feat_nodes, dim=0)  # (1+num_feats, feat_dim_varies)
        
        # Build star edges from central node (0) to each feature node
        edges = [[0, i] for i in range(1, len(feat_nodes) + 1)]
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        
        dummy_graph = Data(x=node_features, edge_index=edge_index)
        return dummy_graph


    def reset_parameters(self):
        # reset params like before
        self.graph_conv1.reset_parameters(); self.graph_bn1.reset_parameters()
        self.graph_conv2.reset_parameters(); self.graph_bn2.reset_parameters()
        self.graph_conv3.reset_parameters(); self.graph_bn3.reset_parameters()
        self.graph_conv4.reset_parameters(); self.graph_bn4.reset_parameters()

        # pool params: if LSTM/GRU was used, reset their parameters if present
        if self.pooling_type == 'lstm':
            try:
                self.pooling.lstm.reset_parameters()
                self.pooling.attention.reset_parameters()
            except Exception:
                pass
        elif self.pooling_type == 'gru':
            try:
                self.pooling.gru.reset_parameters()
                self.pooling.attention.reset_parameters()
            except Exception:
                pass

        try:
            self.node_conv1.reset_parameters()
        except Exception:
            pass
        try:
            self.node_bn1.reset_parameters()
        except Exception:
            pass
        try:
            self.fc1.reset_parameters(); self.fc2.reset_parameters()
        except Exception:
            pass
