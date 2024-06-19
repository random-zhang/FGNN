import math
import multiprocessing

from mrmr import mrmr_classif


import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import spearmanr, entropy,pearsonr
from sklearn.metrics import mutual_info_score,normalized_mutual_info_score
from tqdm import tqdm

import sqlite_util
from MI import CMI



N=2513
K=2500


from sklearn.metrics import mutual_info_score
import numpy as np









task_name='hiv'

# labels = [labels[2]]
label = 'Class'
df=pd.read_csv(f'{task_name}_{label}_fps.csv')
F = df.drop('target', axis=1).values.astype(int)
C = df['target'].values.astype(int)

mic=np.zeros([N])
for i in tqdm(range(N)):
    mic[i] = mutual_info_score(F[:, i], C)
print(mic)

selected_indices = np.argsort(mic)
print(selected_indices)
# 获取元素排序后的索引位置（从大到小）
selected_indices = selected_indices[::-1]
print(selected_indices)
selected_indices=selected_indices.tolist()
for i in range(100,2513,100):

        sqlite_util.createSelectedTable(table_name='mim')
        try:
            sqlite_util.read_from_selectecd(False,task_name, i,label,table_name='mim')
            sqlite_util.updata_selected(task_name, i,str(selected_indices[:i]),label,table_name='mim')
        except Exception:
            sqlite_util.insert_selected(task_name, i,str(selected_indices[:i]),label,table_name='mim')

