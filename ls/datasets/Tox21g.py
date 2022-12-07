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
#from pyg_chemprop_utils import smiles2data
import torch_geometric.utils as tg

from ls.utils.print import print


from typing import List, Union

import numpy as np
import torch.nn as nn
from rdkit import Chem
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch_geometric.data import Data
from tqdm import tqdm


def mol2data(mol):
    atom_feat = [atom_features(atom) for atom in mol.GetAtoms()]

    edge_attr = []
    edge_index = []

    for bond in mol.GetBonds():
        # eid = bond.GetIdx()
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.extend([(i, j), (j, i)])
        b = bond_features(bond)
        edge_attr.extend([b, b.copy()])

    x = torch.FloatTensor(atom_feat)
    edge_attr = torch.FloatTensor(edge_attr)
    edge_index = torch.LongTensor(edge_index).T

    return Data(x=x, edge_attr=edge_attr, edge_index=edge_index)


def smiles2data(smi, explicit_h=True):
    mol = Chem.MolFromSmiles(smi)
    if explicit_h:
        mol = Chem.AddHs(mol)
    return mol2data(mol)




class Tox21g(Dataset):
    def __init__(self):

        #data = pd.read_csv(r"/Users/Rachel/Downloads/tox21.csv")
        data = pd.read_csv(r"tox21.csv")

        targets_df = list(data['NR-AR'])
        smiles_df = list(data['smiles']) 
        
        #filter the lists to not include NaN
        
        targets_lst = []
        smiles_lst = []
        
        for i in range(len(targets_df)):
            if (targets_df[i] == 0) or (targets_df[i] == 1):
                targets_lst.append(targets_df[i])
                smiles_lst. append(smiles_df[i])
            

        self.targets = targets_lst
        self.data = smiles_lst
        
        self.length = len(self.targets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        '''
            Return the molecule representation and the label for the given
            index.
        '''
        return smiles2data(self.data[idx]), torch.tensor(self.targets[idx]).long()
        #return tg.from_smiles(self.data[idx]), torch.tensor(self.targets[idx]).long()
