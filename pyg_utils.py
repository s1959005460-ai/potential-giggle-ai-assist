import torch
import torch.nn as nn

def get_model(config):
    mconf = config.get('model', {})
    mtype = mconf.get('type', 'mlp')
    graph_level = mconf.get('graph_level', True)  # True: graph classification; False: node classification
    if mtype == 'gcn':
        try:
            from torch_geometric.nn import GCNConv, global_mean_pool
            class SimpleGCN(torch.nn.Module):
                def __init__(self, in_channels, hidden, out_channels):
                    super().__init__()
                    self.conv1 = GCNConv(in_channels, hidden)
                    self.conv2 = GCNConv(hidden, hidden)
                    self.lin = nn.Linear(hidden, out_channels)
                    self.pool = global_mean_pool
                def forward(self, data):
                    x, edge_index = data.x, data.edge_index
                    x = torch.relu(self.conv1(x, edge_index))
                    x = torch.relu(self.conv2(x, edge_index))
                    if hasattr(data, 'batch') and graph_level:
                        x = self.pool(x, data.batch)
                    return self.lin(x)
            return SimpleGCN(mconf.get('in_dim', 7), mconf.get('hidden', 64), mconf.get('out', 2))
        except Exception as e:
            print('GCNConv unavailable, fallback to MLP:', e)
    elif mtype == 'gat':
        try:
            from torch_geometric.nn import GATConv, global_mean_pool
            class SimpleGAT(torch.nn.Module):
                def __init__(self, in_channels, hidden, out_channels, heads=4):
                    super().__init__()
                    self.conv1 = GATConv(in_channels, hidden, heads=heads)
                    self.conv2 = GATConv(hidden*heads, hidden, heads=1)
                    self.lin = nn.Linear(hidden, out_channels)
                    self.pool = global_mean_pool
                def forward(self, data):
                    x, edge_index = data.x, data.edge_index
                    x = torch.relu(self.conv1(x, edge_index))
                    x = torch.relu(self.conv2(x, edge_index))
                    if hasattr(data, 'batch') and graph_level:
                        x = self.pool(x, data.batch)
                    return self.lin(x)
            return SimpleGAT(mconf.get('in_dim',7), mconf.get('hidden',64), mconf.get('out',2))
        except Exception as e:
            print('GATConv unavailable, fallback to MLP:', e)
    elif mtype == 'sage':
        try:
            from torch_geometric.nn import SAGEConv, global_mean_pool
            class SimpleSAGE(torch.nn.Module):
                def __init__(self, in_channels, hidden, out_channels):
                    super().__init__()
                    self.conv1 = SAGEConv(in_channels, hidden)
                    self.conv2 = SAGEConv(hidden, hidden)
                    self.lin = nn.Linear(hidden, out_channels)
                    self.pool = global_mean_pool
                def forward(self, data):
                    x, edge_index = data.x, data.edge_index
                    x = torch.relu(self.conv1(x, edge_index))
                    x = torch.relu(self.conv2(x, edge_index))
                    if hasattr(data, 'batch') and graph_level:
                        x = self.pool(x, data.batch)
                    return self.lin(x)
            return SimpleSAGE(mconf.get('in_dim',7), mconf.get('hidden',64), mconf.get('out',2))
        except Exception as e:
            print('SAGEConv unavailable, fallback to MLP:', e)
    elif mtype == 'gin':
        try:
            from torch_geometric.nn import GINConv, global_mean_pool
            class SimpleGIN(torch.nn.Module):
                def __init__(self, in_channels, hidden, out_channels):
                    super().__init__()
                    nn1 = nn.Sequential(nn.Linear(in_channels, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                    self.conv1 = GINConv(nn1)
                    nn2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                    self.conv2 = GINConv(nn2)
                    self.lin = nn.Linear(hidden, out_channels)
                    self.pool = global_mean_pool
                def forward(self, data):
                    x, edge_index = data.x, data.edge_index
                    x = torch.relu(self.conv1(x, edge_index))
                    x = torch.relu(self.conv2(x, edge_index))
                    if hasattr(data, 'batch') and graph_level:
                        x = self.pool(x, data.batch)
                    return self.lin(x)
            return SimpleGIN(mconf.get('in_dim',7), mconf.get('hidden',64), mconf.get('out',2))
        except Exception as e:
            print('GINConv unavailable, fallback to MLP:', e)

    # default MLP fallback (works without PyG)
    in_dim = mconf.get('in_dim', 32)
    hidden = mconf.get('hidden', 64)
    out = mconf.get('out', 10)
    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, out)
    )
    return model
