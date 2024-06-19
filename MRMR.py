import math
import multiprocessing

from mrmr import mrmr_classif

import MRMR
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









task_name='mcf_7'

# labels = [labels[2]]
label = 'Class'
df=pd.read_csv(f'{task_name}_{label}_fps.csv')
F = df.drop('target', axis=1).values.astype(int)
C = df['target'].values.astype(int)

selected_indices = mrmr_classif(X=pd.DataFrame(F), y=pd.Series(C), K=2500)
print(selected_indices)
for i in range(100,2513,100):
        sqlite_util.createSelectedTable(table_name='mrmr')
        try:
            sqlite_util.read_from_selectecd(False,task_name, i,label,table_name='mrmr')
            sqlite_util.updata_selected(task_name, i,str(selected_indices[:i]),label,table_name='mrmr')
        except Exception:
            sqlite_util.insert_selected(task_name, i,str(selected_indices[:i]),label,table_name='mrmr')

