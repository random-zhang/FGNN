import math
import multiprocessing
import sqlite3
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import spearmanr, entropy,pearsonr
from sklearn.metrics import mutual_info_score,normalized_mutual_info_score
from tqdm import tqdm

import sqlite_util
from MI import CMI


def NMI(A,B):
    # 样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    # 互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)    # 输出满足条件的元素的下标
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)   # Find the intersection of two arrays.
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
        Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    if Hx+Hy==0:
        MIhat= np.nan
    else:
        MIhat = 2.0*MI/(Hx+Hy)
    return MIhat

N=2513
K=2500







def calculate_cmi_wrapper(process,chunk ,start, N,F,result_dict):

    mis=[]
    ps=[]
    cmis=[]
    for i in tqdm(range(chunk)):
        mi=[]
        cmi=[]
        p=[]
        for j in range(N):
            if j==i:
                mi.append(np.nan)
                cmi.append(np.nan)
                p.append(np.nan)
            else:
                mi.append(mutual_info_score(F[:, i + start], F[:, j]))
                cmi.append(CMI(F[:, i + start], C,F[:, j]))
        mis.append(mi)
        cmis.append(cmi)
        ps.append(p)


    result_dict[str(process)]=(mis,cmis,ps)

task_name='bbbp'
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
    mic[i] = NMI(F[:, i], C)
selected_indices = [np.nanargmax(mic)]


num_cores = multiprocessing.cpu_count()
processes = []
num_cores=num_cores-3
chunk = math.ceil(N / num_cores)
mis=[]
cmis=[]
ps=[]
manager = multiprocessing.Manager()
result_dict = manager.dict()
for i in range(num_cores):
    start=i*chunk
    if i==num_cores-1:
       chunk=N-i*chunk
    process = multiprocessing.Process(target=calculate_cmi_wrapper, args=(i, chunk, start,N, F,result_dict))
    process.start()
    processes.append(process)



print('dd')
# 等待所有进程完成
for process in processes:
    process.join()
print('rr')

for i in range(num_cores):
     m,c,p=result_dict[str(i)]
     mis.extend(m)
     cmis.extend(c)
     ps.extend(p)

mis=np.array(mis)
cmis=np.array(cmis)
ps=np.array(ps)
total=[i for i in range(N)]

for i in tqdm(range(1,K )):
    M = np.setdiff1d(total, selected_indices)
    # Imax=np.nanmean(mis[M][:,selected_indices],axis=1)
    Imax = np.nanmin(mis[M][:, selected_indices], axis=1)
    # Imax = np.nanmean(m2[M][:,selected_indices],axis=1)
    Icmis = np.nanmin(cmis[M][:, selected_indices], axis=1)
    # Icmis=mic[M]

    # print(Imax.shape,Icmis.shape)
    # selected_indices.append(M[np.nanargmax(Icmis-Imax)])
    selected_indices.append(M[np.nanargmax(Icmis - Imax)])


    if (i+1)%100==0 :
        print(selected_indices)
        sqlite_util.createSelectedTable('test')
        try:
            sqlite_util.read_from_selectecd(False,task_name, i+1,label,'test')
            sqlite_util.updata_selected(task_name, i+1,str(selected_indices),label,'test')
        except Exception:
            sqlite_util.insert_selected(task_name, i+1,str(selected_indices),label,'test')
