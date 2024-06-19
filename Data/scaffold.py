from collections import defaultdict
import random

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from Data.Dataset import MixDataset


def generate_scaffold(mol, include_chirality=False):
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(mol, use_indices=False):
    scaffolds = defaultdict(set)
    for i, one in enumerate(mol):
        scaffold = generate_scaffold(one)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(one)

    return scaffolds


def scaffold_split(all_data,indices, args):
    assert sum(args.split) == 1
    # Split
    train_size, val_size, test_size = int(args.split[0] * len(indices)), int(args.split[1] * len(indices)), int(args.split[2] * len(indices))
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    data=all_data[indices]
    mols=[]
    for one in data:
        mols.append(one.mol)
    scaffold_to_indices = scaffold_to_smiles(mols, use_indices=True)

    index_sets = list(scaffold_to_indices.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    # big_index_sets=np.array(big_index_sets)
    # small_index_sets=np.array(small_index_sets)
    # np.random.shuffle(big_index_sets)
    # np.random.shuffle(small_index_sets)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    #index_sets = big_index_sets + small_index_sets
    index_sets=np.concatenate((big_index_sets,small_index_sets))

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    # Map from indices to data
    # train = [data[i] for i in train]
    # val = [data[i] for i in val]
    # test = [data[i] for i in test]

    return train, val,test
import numpy as np

import numpy as np

def scaffold_split_sampling(all_data, indices, args):
    assert sum(args.split) == 1

    # 分割数据
    train_size, val_size, test_size = int(args.split[0] * len(indices)), int(args.split[1] * len(indices)), int(args.split[2] * len(indices))
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # 将分子与数据索引映射
    data = all_data[indices]
    mols = [one.mol for one in data]
    scaffold_to_indices = scaffold_to_smiles(mols, use_indices=True)

    index_sets = list(scaffold_to_indices.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)

    big_index_sets = np.array(big_index_sets)
    small_index_sets = np.array(small_index_sets)
    np.random.shuffle(big_index_sets)
    np.random.shuffle(small_index_sets)

    # 对训练集进行下采样
    downsampled_train_size = min(train_size, len(big_index_sets))
    downsampled_train_size = 5000
    big_index_sets = big_index_sets[:downsampled_train_size]

    # 连接选择的集合
    index_sets = np.concatenate((big_index_sets, small_index_sets))

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    return train, val, test

