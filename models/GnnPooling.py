##### GAT-LSTM-GIN
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINConv, BatchNorm, GlobalAttention, GINEConv
from torch_geometric.data import Data, Batch

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


############### GIN-GAT ###############
class GINGAT(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_channels, out_channels, heads, dropout, pooling_type, num_tasks):
        super(GINGAT, self).__init__()

        # Define graph-level GNN layers with GINEConv (with edge attributes)
        self.graph_conv1 = GINEConv(nn.Sequential(nn.Linear(node_dim, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)), edge_dim=edge_dim)
        self.graph_bn1 = BatchNorm(hidden_channels)

        self.graph_conv2 = GINEConv(nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)), edge_dim=edge_dim)
        self.graph_bn2 = BatchNorm(hidden_channels)

        self.graph_conv3 = GINEConv(nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)), edge_dim=edge_dim)
        self.graph_bn3 = BatchNorm(hidden_channels)

        self.graph_conv4 = GINEConv(nn.Sequential(nn.Linear(hidden_channels, out_channels), nn.ReLU(), nn.Linear(out_channels, out_channels)), edge_dim=edge_dim)
        self.graph_bn4 = BatchNorm(out_channels)

        # Pooling
        self.pooling_type = pooling_type
        if pooling_type == 'lstm':
            self.pooling = LSTMAttentionPooling(out_channels, out_channels)
        elif pooling_type == 'gru':
            self.pooling = GRUAttentionPooling(out_channels, out_channels)
        elif pooling_type == 'attention':
            self.pooling = GlobalAttention(gate_nn=nn.Linear(out_channels, 1))
        else:
            raise ValueError("please use lstm, gru or attention for pooling")

        # Define node-level GNN with GATv2Conv (without edge attributes) and fully connected layers
        self.node_conv1 = GATv2Conv(out_channels, hidden_channels, heads=heads, concat=False)
        self.node_bn1 = BatchNorm(hidden_channels)

        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_tasks)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch, data):
        device = x.device
        edge_attr = edge_attr.float().to(device)

        # Graph-level GNN
        x = self.graph_conv1(x, edge_index, edge_attr)
        x = self.graph_bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.graph_conv2(x, edge_index, edge_attr)
        x = self.graph_bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.graph_conv3(x, edge_index, edge_attr)
        x = self.graph_bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.graph_conv4(x, edge_index, edge_attr)
        x = self.graph_bn4(x)
        x = F.relu(x)

        # GRU-based pooling with attention
        graph_out = self.pooling(x, batch)

        # Extract fingerprints and descriptors from data
        fingerprints = [data.ECFP.to(device), data.Topological.to(device), data.MACCS.to(device), data.EState.to(device)]
        descriptors = [data.Rdkit2D.to(device), data.Phar2D.to(device)]

        features = fingerprints + descriptors

        # Ensure all features are 2D tensors and concatenate with graph_out
        features = [f.unsqueeze(0) if f.dim() == 1 else f for f in features]

        # Create dummy graphs for the entire batch
        dummy_graphs = []
        for i in range(graph_out.size(0)):
            dummy_graph = self.create_dummy_graph(graph_out[i].unsqueeze(0), [f[i].unsqueeze(0) for f in features], device)
            dummy_graphs.append(dummy_graph)

        # Batch all dummy graphs into a single Batch object
        batched_dummy_graph = Batch.from_data_list(dummy_graphs).to(device)

        # Node-level GNN on the batched dummy graph
        x, edge_index = batched_dummy_graph.x, batched_dummy_graph.edge_index

        x = self.node_conv1(x, edge_index)
        x = self.node_bn1(x)
        x = F.relu(x)

        # Collect predictions from the central nodes
        central_node_indices = torch.arange(0, len(dummy_graphs) * (len(features) + 1), len(features) + 1, device=device)
        x = x[central_node_indices]

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        predictions = self.fc2(x)

        return predictions

    def create_dummy_graph(self, graph_embedding, features, device):
        features = [f.unsqueeze(0).to(device) if f.dim() == 1 else f.to(device) for f in features]
        central_node_features = graph_embedding.to(device)
        node_features = torch.cat([central_node_features] + features, dim=0)
        edges = [[0, i] for i in range(1, len(features) + 1)]
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        dummy_graph = Data(x=node_features, edge_index=edge_index)
        return dummy_graph

    def reset_parameters(self):
        self.graph_conv1.reset_parameters()
        self.graph_bn1.reset_parameters()

        self.graph_conv2.reset_parameters()
        self.graph_bn2.reset_parameters()

        self.graph_conv3.reset_parameters()
        self.graph_bn3.reset_parameters()

        self.graph_conv4.reset_parameters()
        self.graph_bn4.reset_parameters()

        if self.pooling_type == 'lstm':
            self.pooling.lstm.reset_parameters() 
            self.pooling.attention.reset_parameters() 

        elif self.pooling_type == 'gru':
            self.pooling.gru.reset_parameters() 
            self.pooling.attention.reset_parameters() 

        elif self.pooling_type == 'attention':
            self.pooling.reset_parameters()

        self.node_conv1.reset_parameters()
        self.node_bn1.reset_parameters()
        
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

############### GAT-GIN ###############
class GATGIN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_channels, out_channels, heads, dropout, pooling_type, num_tasks):
        super(GATGIN, self).__init__()

        # Define graph-level GNN layers with GATv2Conv (with edge attributes)
        self.graph_conv1 = GATv2Conv(node_dim, hidden_channels, heads=heads, edge_dim=edge_dim, concat=False)
        self.graph_bn1 = BatchNorm(hidden_channels)

        self.graph_conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, edge_dim=edge_dim, concat=False)
        self.graph_bn2 = BatchNorm(hidden_channels)

        self.graph_conv3 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, edge_dim=edge_dim, concat=False)
        self.graph_bn3 = BatchNorm(hidden_channels)

        self.graph_conv4 = GATv2Conv(hidden_channels, out_channels, heads=heads, edge_dim=edge_dim, concat=False)
        self.graph_bn4 = BatchNorm(out_channels)

        # Pooling
        self.pooling_type = pooling_type
        if pooling_type == 'lstm':
            self.pooling = LSTMAttentionPooling(out_channels, out_channels)
        elif pooling_type == 'gru':
            self.pooling = GRUAttentionPooling(out_channels, out_channels)
        elif pooling_type == 'attention':
            self.pooling = GlobalAttention(gate_nn=nn.Linear(out_channels, 1))
        else:
            raise ValueError("please use lstm, gru or attention for pooling")


        # Define node-level GNN with GIN (without edge attributes) and fully connected layers
        self.node_conv1 = GINConv(nn.Sequential(nn.Linear(out_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)))
        self.node_bn1 = BatchNorm(hidden_channels)

        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_tasks)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch, data):
        device = x.device
        edge_attr = edge_attr.float().to(device)

        # Graph-level GNN
        x = self.graph_conv1(x, edge_index, edge_attr)
        x = self.graph_bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.graph_conv2(x, edge_index, edge_attr)
        x = self.graph_bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.graph_conv3(x, edge_index, edge_attr)
        x = self.graph_bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.graph_conv4(x, edge_index, edge_attr)
        x = self.graph_bn4(x)
        x = F.relu(x)

        # LSTM and Attention-based pooling
        graph_out = self.pooling(x, batch)

        # Extract fingerprints and descriptors from data
        fingerprints = [data.ECFP.to(device), data.Topological.to(device), data.MACCS.to(device), data.EState.to(device)]
        descriptors = [data.Rdkit2D.to(device), data.Phar2D.to(device)]

        features = fingerprints + descriptors

        # Ensure all features are 2D tensors
        features = [f.unsqueeze(0) if f.dim() == 1 else f for f in features]

        # Create dummy graphs for the entire batch
        dummy_graphs = []
        for i in range(graph_out.size(0)):
            dummy_graph = self.create_dummy_graph(graph_out[i].unsqueeze(0), [f[i].unsqueeze(0) for f in features], device)
            dummy_graphs.append(dummy_graph)

        # Batch all dummy graphs into a single Batch object
        batched_dummy_graph = Batch.from_data_list(dummy_graphs).to(device)

        # Node-level GNN on the batched dummy graph
        x, edge_index = batched_dummy_graph.x, batched_dummy_graph.edge_index

        x = self.node_conv1(x, edge_index)
        x = self.node_bn1(x)
        x = F.relu(x)

        # Collect predictions from the central nodes
        central_node_indices = torch.arange(0, len(dummy_graphs) * (len(features) + 1), len(features) + 1, device=device)
        x = x[central_node_indices]

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        predictions = self.fc2(x)

        return predictions

    def create_dummy_graph(self, graph_embedding, features, device):
        features = [f.unsqueeze(0).to(device) if f.dim() == 1 else f.to(device) for f in features]
        central_node_features = graph_embedding.to(device)
        node_features = torch.cat([central_node_features] + features, dim=0)
        edges = [[0, i] for i in range(1, len(features) + 1)]
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        dummy_graph = Data(x=node_features, edge_index=edge_index)
        return dummy_graph

    def reset_parameters(self):
        self.graph_conv1.reset_parameters()
        self.graph_bn1.reset_parameters()

        self.graph_conv2.reset_parameters()
        self.graph_bn2.reset_parameters()

        self.graph_conv3.reset_parameters()
        self.graph_bn3.reset_parameters()

        self.graph_conv4.reset_parameters()
        self.graph_bn4.reset_parameters()

        if self.pooling_type == 'lstm':
            self.pooling.lstm.reset_parameters() 
            self.pooling.attention.reset_parameters() 

        elif self.pooling_type == 'gru':
            self.pooling.gru.reset_parameters() 
            self.pooling.attention.reset_parameters() 

        elif self.pooling_type == 'attention':
            self.pooling.reset_parameters()

        self.node_conv1.reset_parameters()
        self.node_bn1.reset_parameters()

        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
