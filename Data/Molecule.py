import random

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from torch.utils.data.dataset import Dataset

from Data.Feature import atom_featurizer, bond_featurizer
from Data.pubchemfp import GetPubChemFPs


atom_type_max = 100
atom_f_dim = 133
atom_features_define = {
    'atom_symbol': list(range(atom_type_max)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'charity_type': [0, 1, 2, 3],
    'hydrogen': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],}


def get_atom_features_dim():
    return atom_f_dim


class Mole:
    def __init__(self, smile, label=None,mol=None,name=None):
        self.smile = smile
        if mol==None:
            mol = Chem.MolFromSmiles(self.smile)
        self.name=name
        if label!=None:
           self.label = np.asarray(label)  # 转换成float
        else:
            self.label=label
        self.mol=mol

    def loader(self,args,i):

        if self.mol==None:
            self.mol = None
        else:

            padding_n=args.max_atom-len(self.mol.GetAtoms())  if  args.max_atom>=len(self.mol.GetAtoms())  else 0
            adj = Chem.rdmolops.GetAdjacencyMatrix(self.mol)
            atom_num = self.mol.GetNumAtoms()
            weight = np.zeros([atom_num, atom_num, args.gat_bonds_input_dim])  # (atom_num,atom_num,7)
            atom_features = []

            for atom in self.mol.GetAtoms():
                atom_feature = atom_featurizer.encode(atom)
                atom_features.append(atom_feature)
                for neighbor in atom.GetNeighbors():
                    bond = self.mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                    f = bond_featurizer.encode(bond)
                    weight[atom.GetIdx(), neighbor.GetIdx()] = f

            fp = []
            fp_maccs = AllChem.GetMACCSKeysFingerprint(self.mol)

            fp_phaErGfp = AllChem.GetErGFingerprint(self.mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
            fp_pubcfp = GetPubChemFPs(self.mol)
            fp_morgan = AllChem.GetMorganFingerprintAsBitVect(self.mol, 2, nBits=1024)
            if args.fp_dim==1024:
                fp.extend(fp_morgan)
            elif args.fp_dim==2513-1024:
                fp.extend(fp_maccs)
                fp.extend(fp_phaErGfp)
                fp.extend(fp_pubcfp)
            else:
                fp.extend(fp_maccs)
                fp.extend(fp_phaErGfp)
                fp.extend(fp_pubcfp)
                fp.extend(fp_morgan)
            fp=np.asarray(fp)
            adj=np.asarray(adj)
            atom_features=np.asarray(atom_features)
            if padding_n > 0:
                atom_features = np.pad(atom_features, ((0, padding_n),(0,0)), mode='constant')
                weight = np.pad(weight, ((0, padding_n), (0, padding_n),(0,0)), mode='constant')
                adj = np.pad(adj, ((0, padding_n), (0, padding_n)), mode='constant')
            else:
                atom_features = atom_features[:args.max_atom,:]
                weight = weight[:args.max_atom, :args.max_atom,:]
                adj = adj[:args.max_atom, :args.max_atom]
            self.atom_features=atom_features
            self.weight =weight
            self.fp=fp
            self.adj = adj
            self.mol = self.mol
            self.no=i


class MoleDataSet(Dataset):
    def __init__(self, data):
        self.data = data


    def smile(self):
        smile_list = []
        for one in self.data:
            smile_list.append(one.smile)
        return smile_list

    def mol(self):
        mol_list = []
        for one in self.data:
            mol_list.append(one.mol)
        return mol_list

    def label(self):
        label_list = []
        for one in self.data:
            label_list.append(one.label)
        return label_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
