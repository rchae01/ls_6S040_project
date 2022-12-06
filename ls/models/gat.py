from .build import ModelFactory

import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv, global_add_pool


@ModelFactory.register('gat')
class Net(torch.nn.Module):
    def __init__(self, include_label: int, 
                 num_classes: int = 2):
        
        super().__init__()
        
        self.conv1 = GATConv(133, 8, heads=8,
                             dropout=0.6).jittable()

        self.conv2 = GATConv(64, 2, heads=1, concat=True,
                             dropout=0.6).jittable()
        
        self.include_label = include_label
        
        self.seq = nn.Sequential(*modules)

    def forward(self, data, y=None):
        
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        #print(x.shape)
        
        
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        
        x = global_add_pool(x, batch)
        
        
        if self.include_label:
            x = torch.cat(
                [x, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )
            

        x = self.seq(x)
        
        
        #return F.log_softmax(x, dim=1)
        return x

