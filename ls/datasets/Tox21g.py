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
       

#         for smile in smiles_df.values:
#             smile = smile[0]
#             smile_data = smiles2data(smile)
#             smiles_lst.append(smile_data)

        self.targets = torch.tensor(targets_df.values)
        self.data = smiles_df
        
        self.length = len(self.targets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        '''
            Return the molecule representation and the label for the given
            index.
        '''
        return self.data[idx], self.targets[idx]
