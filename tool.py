import configparser
import copy
import multiprocessing
import shutil
import os
from datetime import datetime

import math
import random

import numpy as np
import pandas as pd
import torch
from mrmr import mrmr_classif

from sklearn.feature_selection import mutual_info_classif, SelectKBest,chi2
from torch import nn
from tqdm import tqdm

import MI
import sqlite_util
from Data.Molecule import Mole, MoleDataSet
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, mean_squared_error, mutual_info_score
import heapq

from Data.scaffold import scaffold_split
from Model.GQNN import GQModel


def create_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)


def load_data(args,predict_path=None,num=-1):
    if not predict_path:
       df = pd.read_csv(args.data_path,encoding_errors='ignore')
    else:
        df = pd.read_csv(predict_path, encoding_errors='ignore')
    if num!=-1:
       df=df.head(num)
    # df = df.dropna()
    data = []
    if args.mode=='train':
        max_atoms=[]
    if args.mode=='validation' or args.mode=='train':

        for i, row in df.iterrows():
            if len(args.labels)==1 and  math.isnan(row[args.labels[0]]):
               continue
            labels=[row[key] for key in args.labels]

            mol = Mole(row['Smiles'],labels)  # 包装成对象
            data.append(mol)
    else:
        for i, row in df.iterrows():
                if row['Smiles']==None:
                    continue
                try:
                     mol = Mole(row['Smiles'],name=row['name'])  # 包装成对象
                except Exception:

                     mol = Mole(row['Smiles'])  # 包装成对象

                data.append(mol)
    total = len(data)
    valid = []
    for i in range(total):

        if data[i].mol is not None and data[i].mol.GetNumAtoms()<=100:
            if args.mode=='train':
                max_atoms.append(data[i].mol.GetNumAtoms())
            valid.append(data[i])
    valid_num = len(valid)

    if  args.max_atom<np.array(max_atoms).max():
        args.max_atom=np.array(max_atoms).max()


    print('max_atom', args.max_atom)
    for i in tqdm(range(len(valid))):
        valid[i].loader(args,i)
    #print('There are ', valid_num, ' smiles in total.')
    if total - valid_num > 0:
        print('There are ', total, ' smiles first, but ', total - valid_num,
              ' smiles is invalid.  ')

    return valid


def load_model(pred_args):
    state = torch.load(pred_args.model_path, map_location=lambda storage, loc: storage)
    # args = state['args']

    # if pred_args is not None:
    #     for key, value in vars(pred_args).items():
    #         if not hasattr(args, key):
    #             setattr(args, key, value)

    state_dict = state['state_dict']
    model = GQModel(pred_args)
    model_state_dict = model.state_dict()

    load_state_dict = {}
    for param in state_dict.keys():
        if param not in model_state_dict:
            print(f'Parameter is not found: {param}.')
        elif model_state_dict[param].shape != state_dict[param].shape:
            print(f'Shape of parameter is error: {param}.')
        else:
            load_state_dict[param] = state_dict[param]
            # print(f'Load parameter: {param}.')

    model_state_dict.update(load_state_dict)
    model.load_state_dict(model_state_dict)

    model = model.to(pred_args.device)

    return model





def save_model(path, model, args):
    state = {
        'args': args,
        'state_dict': model.state_dict(),
        'data_scaler': None
    }

    torch.save(state, path)


def load_args(path):
    state = torch.load(path, map_location=lambda storage, loc: storage)

    return state['args']
def compute_score(label,pred, args):
    # pred=torch.Tensor(pred)
    # label=torch.Tensor(label)
    result=None
    if args.dataset_type=='classification':
        if args.classification_type=='ROC':
            label=np.array(label)
            pred=np.array(pred)
            result = roc_auc_score(label, pred)
        else:
            precision, recall, thresholds = precision_recall_curve(np.array(label), np.array(pred))
            result = auc(recall, precision)
    else:
        result = np.sqrt(mean_squared_error(np.array(label), np.array(pred)))


    return result
def predict(model, dataloader, args):
    model.eval()
    pred = []

    for atom_feature, weight, adj, fp, labels ,smiles,no in dataloader:

        with torch.no_grad():
            pred_now =model(atom_feature.to(args.device).float(),weight.to(args.device).float(),adj.to(args.device),fp.to(args.device).float(),smiles,no,labels)

        pred_now = pred_now.data.cpu().numpy()
        pred_now = np.array(pred_now).astype(float)

        pred.extend(pred_now)
    pred=[x for x in pred]

    return pred

def get(rate,pred,target):
        if rate >1:
           values=list(enumerate(pred))
           values=heapq.nlargest(rate,values,key=lambda x:x[1])
           pre=[0 for _ in range(len(pred))]
           for index,_ in values:
               pre[index]=1

        else:
           pre = [1 if x > rate else 0 for x in pred]
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for p, t in zip(pre, target):
            if p == 1 and t == 1:
                TP += 1
            elif p == 0 and t == 0:
                TN += 1
            elif p == 1 and t == 0:
                FP += 1
            else:
                FN += 1
        try:
            Recall = TP / (TP + FN)
        except Exception:
            Recall=0
        try:
            Precision = TP / (TP + FP)
        except Exception:
            Precision = 0

        try:
            F1score=(2 * (Precision * Recall)) / (Precision + Recall)
        except Exception:
            F1score=0
        try:
            F0_5score = ((1+(0.5)**2)*Precision * Recall)/((0.5**2)*Precision + Recall)
        except Exception:
            F0_5score =0

        # print("rate:{} Recall: {:.8f}".format(rate, Recall), end=' ')
        # print("Precision: {:.8f}".format(Precision), end=' ')
       # MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        # print("MCC: {:.8f}".format(MCC), end=' ')
        return Recall,Precision,F1score,F0_5score
def metric(pred,target):
        rate=0.1
        for i in range(1, 10):
            get(i*rate,pred,target)
        get(0.99,pred,target)
        get(100, pred, target)
        print()
def write(ratio, pred, target,writer,task,fold_iter,iter):
    recall, precision, F1score, F0_5score = get(ratio, pred, target)
    if ratio<1:
        writer.add_scalars(f"Recall_{ratio}_{fold_iter}", {task: recall}, iter)
        writer.add_scalars(f"Precision_{ratio}_{fold_iter}", {task: precision}, iter)
        writer.add_scalars(f"F1Score_ratio{ratio}_{fold_iter}", {task: F1score}, iter)
        writer.add_scalars(f"F0.5Score_{ratio}_{fold_iter}", {task: F0_5score}, iter)
    else:
        writer.add_scalars(f"Recall_top_{ratio}_{fold_iter}", {task: recall}, iter)
        writer.add_scalars(f"Precision_top{ratio}_{fold_iter}", {task: precision}, iter)
        writer.add_scalars(f"F1Score_top{ratio}_{fold_iter}", {task: F1score}, iter)
        writer.add_scalars(f"F0.5Score_top{ratio}_{fold_iter}", {task: F0_5score}, iter)
def metrics(task,writer,pred,target,fold_iter,iter):
    rate = 0.1
    for i in range(1, 10):
        write(i * rate, pred, target,writer,task,fold_iter,iter)
    write(0.99, pred, target, writer, task,fold_iter, iter)
    write(100, pred, target, writer, task,fold_iter, iter)
    write(500, pred, target, writer, task,fold_iter, iter)

def one_hot_encoding(index,length):
    encoding = [0] * (length+ 1)
    if index<length:
        encoding[index] = 1
    else:
        encoding[-1]=1
    return encoding
def gettime():
    current_time = datetime.now()

    # 格式化时间作为文件名
    file_name = current_time.strftime("%Y%m%d_%H%M%S")

    return file_name


def add2ini(filename,title,dict):
    config = configparser.ConfigParser()

    # 添加配置项
    config[title] =dict
    # 写入配置文件
    with open(f"{filename}.ini", "a") as config_file:
        config.write(config_file)
def deleteini(filename):
    os.remove(f"{filename}.ini")
def read_ini(filename,title,key):
    config = configparser.ConfigParser()
    config.read(f"{filename}.ini")
    for section in config.sections():
        if title == section:
            for k, value in config.items(title):
                if key ==k:
                    return value
    return None
def update_ini(filename, section, option, value):
    config = configparser.ConfigParser()
    config.read(f"{filename}.ini")

    # 更新配置项的值
    config.set(section, option, str(value))

    # 写入配置文件
    with open(f"{filename}.ini", "w") as config_file:
        config.write(config_file)
def is_option_empty(filename, section, option):
    config = configparser.ConfigParser()
    config.read(f"{filename}.ini")
    try:
        # 获取配置项的值
        value = config.get(section, option)
        # 判断值是否为空
        if value:
            return False
        else:
            return True
    except Exception as e:
        return True

def setSeed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def my_score_func(X, y):
    X=X.astype(int)
    y=y.astype(int)
    mis=[]

    print()
    for i in range(X.shape[1]):
        mi=MI.NMI(X[:,i], y)
        mis.append(mi)
    mis=np.asarray(mis)
    return mis
def feature_selected(datas,args,):

    if args.fp_dim != 2513 and  args.fp_dim!=1024 and args.fp_dim!=2513-1024 and args.fp_dim!=0:
        # fps = []
        # for one in datas:
        #     fps.append(one.fp)
        # fps = np.asarray(fps)


        selected_indices = sqlite_util.read_from_selectecd(args.sorted,args.task_name, args.fp_dim,args.labels[0],args.select_method)
        # elif   args.select_method=='mi':
        #
        #    label_list = [one.label[0] for one in datas]
        #    label_list=np.asarray(label_list)
        #    mis=my_score_func(fps,label_list)
        #    top_K_indices = np.argsort(-mis)[:args.fp_dim]
        #    #selected_indices=sorted(top_K_indices)
        #    selected_indices=top_K_indices
 #        print((np.array(selected_indices)[[441, 531, 171,  14  , 3 ,213 ,  8 ,575 , 34 ,614 ,  4,  56, 119, 491,   6 ,  2 ,415,  81
 # ,474 ,510, 601 ,143, 108, 234 ,463, 588, 570, 112 , 79 ,223]]).tolist())
        for i in range(len(datas)):
            datas[i].fp = datas[i].fp[selected_indices]
    return datas
# def undersampling(d,args):
#     # d=copy.deepcopy(data)
#     if len(d)<10000 or len(d[0].label)!=1:
#         return d
#     labels=[one.label[0] for one in d]
#     labels=np.array(labels)
#     rate=labels.sum()/(len(labels)-labels.sum())
#     p=[]
#     n=[]
#     for one in d:
#         if int(one.label[0])==1:
#             p.append(one)
#         else:
#             n.append(one)
#     q=2
#     if rate>1:
#         m=int(q*len(n))
#         if(m<len(p)):
#             p=p[:m]
#     else :
#         m=int(q*len(p))
#         if m<len(n):
#             n=n[:m]
#     new_data=p
#     new_data.extend(n)
#     random.seed(args.seed)
#     random.shuffle(new_data)
#     print('under sampling :',len(new_data))
#     del d
#     return  new_data
def undersampling(d,indices,args):
    # d=copy.deepcopy(data)
    if len(indices)<10000 or len(args.labels)!=1:
        return indices
    labels=[one.label[0] for one in d[indices]]
    labels=np.array(labels)
    rate=labels.sum()/(len(labels)-labels.sum())
    p=[]
    n=[]
    for i in indices:
        if int(d[i].label[0])==1:
            p.append(i)
        else:
            n.append(i)
    if args.task_name=='muv':
        q=10
    else:
        q=3
        #q=2
    if rate>1:
        m=int(q*len(n))
        if(m<len(p)):
            p=p[:m]
    else :
        m=int(q*len(p))
        if m<len(n):
            n=n[:m]
    new_data=p
    new_data.extend(n)
    random.seed(args.seed)
    random.shuffle(new_data)
    print('under sampling :',len(new_data))

    return  new_data

# def generate_unique_random_numbers(n, count):
#     if count > n:
#         raise ValueError("Count cannot be greater than n.")
#
#     numbers = set()
#     while len(numbers) < count:
#         num = random.randint(0, n)
#         numbers.add(num)
def generate_unique_random_numbers(n, count):
        if count > n:
            raise ValueError("Count cannot be greater than n.")

        numbers = set()
        while len(numbers) < count:
            num = np.random.randint(0, n)
            numbers.add(num)
        return list(numbers)
def add_noise(args,data):
    d=copy.deepcopy(data)
    for i in range(int(args.noise_rate*len(d))):
        d[i].label[0]=1-d[i].label[0]
    # random_index=generate_unique_random_numbers(len(d),int(args.noise_rate*len(d)))
    # for i in random_index:
    #     if i<len(d):
    #        d[i].label[0]=1-d[i].label[0]
    return d


def split_data(data, type, size, seed, log):
    assert len(size) == 3
    assert sum(size) == 1

    if type == 'random':
        data.random_data(seed)
        train_size = int(size[0] * len(data))
        val_size = int(size[1] * len(data))
        train_val_size = train_size + val_size
        train_data = data[:train_size]
        val_data = data[train_size:train_val_size]
        test_data = data[train_val_size:]

        return MoleDataSet(train_data), MoleDataSet(val_data), MoleDataSet(test_data)
    elif type == 'scaffold':  # 将同属性放在一个组里
        return scaffold_split(data, size, seed, log)
    else:
        raise ValueError('Split_type is Error.')
import torch

def adj_to_sparse(adj):
    N = adj.size(0)  # 获取节点数
    E = adj.sum().item()  # 获取边数

    # 构建稀疏矩阵的 indices 和 values
    indices = torch.nonzero(adj).t()
    values = adj[indices[0], indices[1]]

    # 创建稀疏矩阵
    sparse_adj = torch.sparse_coo_tensor(indices, values, size=(N, N))

    # 转置稀疏矩阵
    transposed_adj = sparse_adj.t()

    return transposed_adj

# 示例使用
