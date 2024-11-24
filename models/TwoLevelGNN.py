import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GATv2Conv, global_mean_pool, BatchNorm
from torch_geometric.data import Data, Batch
from torch_geometric.typing import Adj

from torch_geometric.nn.aggr import SortAggregation, Set2Set
from torch_geometric.nn import GlobalAttention



class SimpleTwoLevelGNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_channels, out_channels, heads, dropout):
        super(SimpleTwoLevelGNN, self).__init__()

        # Graph-level GNN with GINEConv
        self.graph_conv1 = GINEConv(torch.nn.Linear(node_dim, hidden_channels), edge_dim=edge_dim)
        self.graph_bn1 = BatchNorm(hidden_channels)

        self.graph_conv2 = GINEConv(torch.nn.Linear(hidden_channels, out_channels), edge_dim=edge_dim)
        self.graph_bn2 = BatchNorm(out_channels)

        # Node-level GNN with GATv2Conv
        self.node_conv1 = GATv2Conv(out_channels, hidden_channels, heads=heads, concat=False)
        self.node_bn1 = BatchNorm(hidden_channels)

        self.node_conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=heads, concat=False)
        self.node_bn2 = BatchNorm(hidden_channels)

        self.fc = torch.nn.Linear(hidden_channels, 1)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch, data):
        edge_attr = edge_attr.float()
        # Graph-level GNN
        x = self.graph_conv1(x, edge_index, edge_attr)
        x = self.graph_bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.graph_conv2(x, edge_index, edge_attr)
        x = self.graph_bn2(x)
        x = F.relu(x)

        # Use mean pooling instead of add pooling
        graph_out = global_mean_pool(x, batch)

        # Extract fingerprints and descriptors from data
        fingerprints = [data.ECFP, data.Topological, data.MACCS, data.EState]
        descriptors = [data.Rdkit2D, data.Phar2D]

        features = fingerprints + descriptors

        # Ensure all features are 2D tensors
        features = [f.unsqueeze(0) if f.dim() == 1 else f for f in features]

        predictions = []
        for i in range(graph_out.size(0)):
            # Dummy graph construction for each graph in the batch
            dummy_graph = self.create_dummy_graph(graph_out[i].unsqueeze(0), [f[i].unsqueeze(0) for f in features])

            # Node-level GNN
            x, edge_index = dummy_graph.x, dummy_graph.edge_index

            x = self.node_conv1(x, edge_index)
            x = self.node_bn1(x)
            x = F.relu(x)
            x = self.dropout(x)

            x = self.node_conv2(x, edge_index)
            x = self.node_bn2(x)
            x = F.relu(x)

            prediction = self.fc(x[0])  # Prediction from the central node
            predictions.append(prediction)

        predictions = torch.stack(predictions, dim=0)
        return predictions

    def create_dummy_graph(self, graph_embedding, features):
        features = [f.unsqueeze(0) if f.dim() == 1 else f for f in features]
        central_node_features = graph_embedding
        node_features = torch.cat([central_node_features] + features, dim=0)
        edges = [[0, i] for i in range(1, len(features) + 1)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        dummy_graph = Data(x=node_features, edge_index=edge_index)
        return dummy_graph


# model_simple = SimpleTwoLevelGNN(node_dim=9, edge_dim=3, hidden_channels=64, out_channels=N_COMPONENTS, heads=8, dropout=0.2)
# optimizer_model_simple = torch.optim.Adam(model_simple.parameters(), lr=0.002, weight_decay=0.0003)

# summary(model_simple)


class SimpleTwoLevelGNNV2(torch.nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_channels: int, out_channels: int, heads: int, dropout: float, num_tasks : int):
        super(SimpleTwoLevelGNNV2, self).__init__()

        # Graph-level GNN with GINEConv
        self.graph_conv1 = GINEConv(nn.Linear(node_dim, hidden_channels), edge_dim=edge_dim)
        self.graph_bn1 = BatchNorm(hidden_channels)

        self.graph_conv2 = GINEConv(nn.Linear(hidden_channels, hidden_channels), edge_dim=edge_dim)
        self.graph_bn2 = BatchNorm(hidden_channels)

        self.graph_conv3 = GINEConv(nn.Linear(hidden_channels, hidden_channels), edge_dim=edge_dim)
        self.graph_bn3 = BatchNorm(hidden_channels)

        self.graph_conv4 = GINEConv(nn.Linear(hidden_channels, out_channels), edge_dim=edge_dim)
        self.graph_bn4 = BatchNorm(out_channels)

        self.global_pooling = GlobalAttention(gate_nn=nn.Linear(out_channels, 1))

        # Node-level GNN with GATv2Conv
        self.node_conv1 = GATv2Conv(out_channels, hidden_channels, heads=heads, concat=False)
        self.node_bn1 = BatchNorm(hidden_channels)

        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_tasks)

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
        

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor, batch: Tensor, data: Data) -> Tensor:

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

        # Use mean pooling instead of add pooling
        # graph_out = global_mean_pool(x, batch)
        graph_out = self.global_pooling(x, batch)


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

    def create_dummy_graph(self, graph_embedding: Tensor, features: list[Tensor], device: torch.device) -> Data:
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
        self.global_pooling.reset_parameters()
        self.node_conv1.reset_parameters()
        self.node_bn1.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
