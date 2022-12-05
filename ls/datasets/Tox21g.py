import os
from os.path import join
import csv
import gzip

from scipy import io
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
import pandas as pd
import torch_geometric.utils as tg
from pyg_chemprop_utils import smiles2data

from ls.utils.print import print


class Tox21g(Dataset):
    def __init__(self):

        data = pd.read_csv(r"/Users/Rachel/Downloads/tox21.csv")

        targets_df = list(data['NR-AR'])
        smiles_df = list(data['smiles'])

        self.targets = targets_df
        self.data = smiles_df
        
        self.length = len(self.targets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        '''
            Return the molecule representation and the label for the given
            index.
        '''
        return smiles2data(self.data[idx]), torch.tensor(self.targets[idx], dtype=torch.long)
