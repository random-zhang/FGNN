import math
import multiprocessing


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







def calculate_cmi_wrapper(process,chunk ,start, N,F,result_dict):

    mis=[]

    cmis1=[]
    cmis2 = []
    for i in tqdm(range(chunk)):
        mi=[]
        cmi1=[]
        cmi2 = []
        for j in range(N):
            if j==i:
                mi.append(np.nan)
                cmi1.append(np.nan)
                cmi2.append(np.nan)
            else:
                mi.append(mutual_info_score(F[:, i + start], F[:, j]))
                cmi1.append(CMI(C,F[:, i + start], F[:, j]))
                cmi2.append(CMI(C, F[:, j],F[:, i + start]))
        mis.append(mi)
        cmis1.append(cmi1)
        cmis2.append(cmi2)
    result_dict[str(process)]=(mis,cmis1,cmis2)

task_name='hiv'
label='Product issues'
labels = ['Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues',
                   'Eye disorders', 'Investigations', 'Musculoskeletal and connective tissue disorders',
                   'Gastrointestinal disorders', 'Social circumstances', 'Immune system disorders',
                   'Reproductive system and breast disorders',
                   'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
                   'General disorders and administration site conditions', 'Endocrine disorders',
                   'Surgical and medical procedures', 'Vascular disorders', 'Blood and lymphatic system disorders',
                   'Skin and subcutaneous tissue disorders', 'Congenital, familial and genetic disorders',
                   'Infections and infestations', 'Respiratory, thoracic and mediastinal disorders',
                   'Psychiatric disorders', 'Renal and urinary disorders',
                   'Pregnancy, puerperium and perinatal conditions', 'Ear and labyrinth disorders',
                   'Cardiac disorders', 'Nervous system disorders',
                   'Injury, poisoning and procedural complications']
# labels = [labels[2]]
label = 'Class'
df=pd.read_csv(f'{task_name}_{label}_fps.csv')
F = df.drop('target', axis=1).values.astype(int)
C = df['target'].values.astype(int)

mic=np.zeros([N])
for i in tqdm(range(N)):
    mic[i] = mutual_info_score(F[:, i], C)
selected_indices = [np.nanargmax(mic)]


num_cores = multiprocessing.cpu_count()
processes = []
num_cores=num_cores-10
chunk = math.ceil(N / num_cores)
mis=[]
cmis1=[]
cmis2=[]

manager = multiprocessing.Manager()
result_dict = manager.dict()
for i in range(num_cores):
    start=i*chunk
    if i==num_cores-1:
       chunk=N-i*chunk
    process = multiprocessing.Process(target=calculate_cmi_wrapper, args=(i, chunk, start,N, F,result_dict))
    process.start()
    processes.append(process)


for process in processes:
    process.join()

for i in range(num_cores):
     m,c1,c2=result_dict[str(i)]
     mis.extend(m)
     cmis1.extend(c1)
     cmis2.extend(c2)

mis=np.array(mis)
cmis1=np.array(cmis1)
cmis2=np.array(cmis2)

total=[i for i in range(N)]


for i in tqdm(range(1,K )):

    M = np.setdiff1d(total, selected_indices)
    J=mic[M]+cmis1[M][:,selected_indices].sum(axis=-1)+cmis2[M][:,selected_indices].sum(axis=-1)
    selected_indices.append(M[np.argmax(J)])
    if (i+1)%100==0 :
        print(selected_indices)
        sqlite_util.createSelectedTable(table_name='mri')
        try:
            sqlite_util.read_from_selectecd(False,task_name, i+1,label,table_name='mri')
            sqlite_util.updata_selected(task_name, i+1,str(selected_indices),label,table_name='mri')
        except Exception:
            sqlite_util.insert_selected(task_name, i+1,str(selected_indices),label,table_name='mri')

