from .build import ModelFactory

import os.path as osp

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.transforms import OneHotDegree

'''
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data', 'TU')
dataset = TUDataset(path, name='IMDB-BINARY', transform=OneHotDegree(135))
dataset = dataset.shuffle()
test_dataset = dataset[:len(dataset) // 10]
train_dataset = dataset[len(dataset) // 10:]
test_loader = DataLoader(test_dataset, batch_size=128)
train_loader = DataLoader(train_dataset, batch_size=128)
'''

@ModelFactory.register('gin')
class Net(torch.nn.Module):
    def __init__(self, include_label: int, 
                 num_classes: int = 2):
        
        super().__init__()
        
        in_channels = 9
        hidden_channels = 9
        out_channels = 2
        num_layers = 3
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.num_classes = num_classes
        
        self.include_label = include_label

        for i in range(num_layers):
            mlp = Sequential(
                Linear(in_channels, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            conv = GINConv(mlp, train_eps=True).jittable()

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))

            in_channels = hidden_channels

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        
        hidden = 2
        self.seq = nn.Linear(hidden + self.include_label, self.num_classes)

    def forward(self, data, y=None):
        
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
        x = global_add_pool(x, batch)
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
       
        
        
        if self.include_label:
            x = torch.cat(
                [x, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )

        x = self.seq(x)
        
        #return F.log_softmax(x, dim=-1)
        return x

'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features, 64, dataset.num_classes, num_layers=3)
model = model.to(device)
model = torch.jit.script(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()

    total_loss = 0.
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.max(dim=1)[1]
        total_correct += pred.eq(data.y).sum().item()
    return total_correct / len(loader.dataset)



for epoch in range(1, 101):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'Train: {train_acc:.4f}, Test: {test_acc:.4f}')
'''

