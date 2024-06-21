import multiprocessing
from datetime import datetime

import math
import shutil

import numpy as np
import os


import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from torch import  nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm



import tool
from Data.Dataset import MixDataset

import random

from Data.scaffold import scaffold_split, scaffold_split_sampling
from Model.GQNN import GQModel
from torch.utils.tensorboard import SummaryWriter

import copy

from paratmeter import add_train_argument, set_hyper_parameter


#525.116.04

def starttrain(args,data=None):
    p = "logs_{}".format(args.task_name)

    args.save_path = 'model_save/{}'.format(args.task_name)
    args.data_path = 'dataset/{}/train.csv'.format(args.task_name)
    args.predict_path=f'result/{args.task_name}'
    args.best_metric = 0

    tool.create_dir(args.save_path)
    if os.path.exists(p):
        shutil.rmtree(p)
    writer = SummaryWriter(log_dir=p, comment='test_tensorboard',
                           filename_suffix='test')  # 建立一个保存数据用的东西，save是输出的文件名
    print('start:', args.task_name,'label:' ,args.labels)

    seeds = [ 10, 20, 30, 40, 50, 60, 70, 80, 90,100]
    if data==None:

       data = tool.load_data(args)  # all data
    if len(args.labels) == 1:
           print('feature_select:',args.fp_dim)
           data = tool.feature_selected(data, args)
    data=np.array(data)
    if args.task_name=='sider':
        np.random.seed(999)
        np.random.shuffle(data)
    graph_path = args.graph_path
    train_avgs=[]
    test_avgs = []

    for seed in seeds:
      # if seed>=90:
        args.graph_path=os.path.join(graph_path,str(seed))
        predict_path=os.path.join(args.predict_path,str(seed))
        if os.path.exists(predict_path):
            shutil.rmtree(predict_path)
        os.makedirs(predict_path)
        if not os.path.exists(args.graph_path):
            os.makedirs(args.graph_path)
        args.seed = seed
        tool.setSeed(args.seed)
        indices=[index for index in range(len(data))]
        indices = np.array(indices)
        args.split=(0.8,0.1,0.1)
        if args.split_type=='random':
            np.random.shuffle(indices)
            # if args.task_name=='muv':
            #     labels = [d.label[0] for d in data[indices]]
            #     train_data, valid_data, y_train, y_valid = train_test_split(indices, labels, test_size=0.8,
            #                                                                     stratify=labels,random_state=args.seed)
            #     val_data, test_data, _, _ = train_test_split(valid_data, y_valid, test_size=0.5, stratify=y_valid,random_state=args.seed)
            # else:
            train_data, val_data, test_data = (
                indices[:int(0.8 * len(indices))], indices[int(0.8 * len(indices)):int(0.9 * len(indices))],
                indices[int(0.9 * len(indices)):])

            # if len(args.labels) == 1 and args.dataset_type=='classification':
            #     labels = [d.label[0] for d in data[indices]]
            #     train_data, valid_data, y_train, y_valid = train_test_split(indices, labels, test_size=0.8,
            #                                                                 stratify=labels,random_state=args.seed)
            #     val_data, test_data, _, _ = train_test_split(valid_data, y_valid, test_size=0.5, stratify=y_valid,random_state=args.seed)
            # else:
            #     np.random.shuffle(indices)
            #     train_data, val_data, test_data = (
            #                 indices[:int(0.8 * len(indices))], indices[int(0.8 * len(indices)):int(0.9 * len(indices))],
            #                 indices[int(0.9 * len(indices)):])
            if args.dataset_type == 'classification' and args.task_name!='muv':
                train_data = tool.undersampling(data, train_data, args)
        else:
            np.random.shuffle(indices)
            train_data, val_data, test_data=     scaffold_split(data,indices, args)
           # train_data, val_data, test_data = scaffold_split_sampling(data, indices, args)
            # train_data = tool.undersampling(data, train_data, args)

        if args.noise_rate!=0:
            train_data=data[train_data]
            val_data=data[val_data]
            train_data = tool.add_noise(args, train_data)
            val_data = tool.add_noise(args, val_data)
            b = seed_train(args, writer,train_data,val_data, args.seed)
        else:
            b = seed_train(args, writer, data[train_data], data[val_data], args.seed)
        train_avgs.append(b)
        path = os.path.join(args.save_path, str(args.seed) , 'model.pt')
        model_args = load_model_args(path)
        model_args.model_path = path
        model = tool.load_model(model_args)
        test_dataset=MixDataset(data[test_data])
        test_dataloader=DataLoader(test_dataset,batch_size=args.batch_size)
        test_pred = tool.predict(model, test_dataloader, model_args)
        test_pred = np.array(test_pred)
        test_pred = test_pred.tolist()
        target = [x.label for x in data[test_data]]
        smiles=[x.smile for x in data[test_data]]
        test_score = tool.compute_score(target, test_pred, model_args)
        target=np.array(target)

        predict_df={'Smiles': smiles}
        if args.labels==1:
            predict_df['Class']=target
        else:
            for i in range(target.shape[1]):
                predict_df[args.labels[i]] = target[:,i]
        df=pd.DataFrame(predict_df)
        df.to_csv(os.path.join(args.save_path,str(seed),'predict.csv'),index=False)
        df['pred']=test_pred
        df.to_csv(os.path.join(predict_path,'result.csv'),index=False)
        print('val auc:', b, 'test auc', test_score)
        test_avgs.append(test_score)
    test_avgs=np.array(test_avgs)
    train_avgs = np.array(train_avgs)
    return test_avgs.mean(),test_avgs.std(),train_avgs.mean(),train_avgs.std()
import numpy as np




def load_model_args(model_path):
    model_args = tool.load_args(model_path)
    return model_args

def epoch_train(model, train_dataloder, optimizer, args):
    torch.cuda.empty_cache()
    args.mode='train'
    model.train()
    loss_sum = 0
    targets = []
    preds = []
    for atom_feature,weight,adj,fp,labels,smiles ,no in train_dataloder:
        target = labels.float().cpu()
        targets.extend(target.numpy())
        pred = model(atom_feature.to(args.device).float(), weight.to(args.device).float(), adj.to(args.device),
                     fp.to(args.device).float(),smiles,no,labels)
        preds.extend(pred.squeeze(axis=1).cpu().tolist())
        if args.dataset_type == 'regression':
            # criterion = nn.SmoothL1Loss()
            criterion = nn.HuberLoss(delta=args.h_delta)
        else:
                criterion = nn.BCEWithLogitsLoss()

        loss = criterion(pred.cpu(), target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    train_score = tool.compute_score(targets, preds, args)
    return loss_sum / len(train_dataloder), train_score, preds, targets


def seed_train(args, writer, train_dataset, val_dataset, seed):
    tool.setSeed(args.seed)
    torch.cuda.empty_cache()
    model = GQModel(args)
    model = model.to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)#, weight_decay=0
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    if args.dataset_type == 'classification':
        best_auc = 0
    else:
        best_auc = math.inf
    aucs = []
    train_dataset=MixDataset(train_dataset)
    val_dataset=MixDataset(val_dataset)
    train_loader=DataLoader(train_dataset,batch_size=args.batch_size)
    val_loder=DataLoader(val_dataset,batch_size=args.batch_size)
    pbar=tqdm(total=args.epochs)
    patience = 7
    count=0
    f_count=0
    best_loss=math.inf
    for epoch in range(args.epochs):
        train_loss, train_auc, preds, targets = epoch_train(model, train_loader, optimizer, args)
        val_pred = tool.predict(model, val_loder, args)
        val_target = [x[-3] for x in val_dataset]
        val_target=torch.Tensor(np.array(val_target)).cpu()
        val_pred=torch.Tensor(np.array(val_pred)).cpu()
        if args.dataset_type == 'classification':
                criterion = nn.BCELoss()


        else:
           # criterion = nn.SmoothL1Loss()
            criterion = nn.HuberLoss(delta=args.h_delta)
        val_loss = criterion(val_pred,val_target)
        if len(args.labels)==1:
           val_score = tool.compute_score(val_target, val_pred, args)
        else:
            val_score = roc_auc_score(val_target, val_pred, average='macro')
        lr_scheduler.step(val_loss)
        writer.add_scalars(f"Loss_{seed}", {"Train": train_loss, 'Valid': val_loss}, epoch)
        writer.add_scalars(f"AUC_{seed}", {"Train": train_auc, "Valid": val_score}, epoch)
        aucs.append(val_score)
        if best_loss > val_loss:
            best_loss = val_loss
            best_auc = val_score
            count = 0

            path = os.path.join(args.save_path, str(seed))
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            tool.save_model(os.path.join(path, 'model.pt'), model, args)
        else:
            count += 1
            if count > patience:
                break
        if val_score==0.5:
            if f_count==5:
                return 0.5
            else:
                f_count+=1
        else:
            f_count=0
        if args.dataset_type == 'classification' and val_score == 0.5 and count > 3:
            return 0.5

        pbar.update(1)
        pbar.set_postfix({'seed': seed, 'epoch': str(epoch + 1), 'AUC': val_score,'loss': val_loss,'best_loss':best_loss},refresh=True)
    pbar.close()
    return best_auc





def fp_2513(args):

    print('fp_dim:',args.fp_dim)
    print('ratio:', args.ratio)
    metric = starttrain(args)
    print(metric)

if __name__ == '__main__':
    p = add_train_argument()
    args = p.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.mode = 'train'
    print(args.device)
    if  os.path.exists(args.graph_path):
        shutil.rmtree(args.graph_path)


    torch.backends.cudnn.benchmark = False  # 模型卷积层预先优化关闭

    torch.backends.cudnn.deterministic = True  # 确定为默认卷积算法
    args=set_hyper_parameter(args)
    stds=[]
    avgs=[]


    # args.select_method='mixed'
    args.with_weight = True
    #(0.8862940343585256, 0.031852305942277614, 0.9035317514475556, 0.016962176457735994)
    args.with_gci= True
    args.with_rnn = True
    args.with_fc = False
    print(args.fp_dim)
    print('noise',args.noise_rate)
    print('select_mode',args.select_method)
    print('split type:',args.split_type)
    print('with_gci',args.with_gci)
    args.sorted = False
    #FGMNN 2513

    #args.fp_dim = 700
    # fp_2513(args)

    # #GNN
    #
    #args.fp_dim=500
    # args.ratio = 0.0
    # args.ratio=0.3
    #args.fp_dim = 700
    #args.ratio = 0.1

    fp_2513(args)


    # #FPN 1800
    # args.fp_dim = 600
    # args.ratio = 0.0
    # fp_2513(args)
    # FPN 2513
    # args.fp_dim = 2513
    # args.ratio = 0.0
    # fp_2513(args)

    # fp_seletor(args)









