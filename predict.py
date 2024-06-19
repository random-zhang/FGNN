import copy
import os
from argparse import ArgumentParser

import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch_geometric.explain as explain
import pandas as pd

from Data.Dataset import MixDataset
from tool import *
from train import load_model_args
def predict(model, dataloader, args):
    model.eval()
    pred = []

    for atom_feature, weight, adj, fp, labels ,smiles,no in dataloader:
        # with torch.no_grad():
            pred_now = model(atom_feature.to(args.device).float(), weight.to(args.device).float(), adj.to(args.device),
                             fp.to(args.device).float(), smiles, no, labels)

            pred_now = pred_now.data.cpu().numpy()
            pred_now = np.array(pred_now).astype(float)

            pred.extend(pred_now)

    pred=[x for x in pred]

    return pred

def p(task, seed):
    p = ArgumentParser()
    p.add_argument('--model_path', type=str, default=f'model_save/{task}/{seed}/model.pt')
    p.add_argument('--predict_path', type=str, default=f'model_save/{task}/{seed}/predict.csv')
    p.add_argument('--graph_path', type=str, default=f'result/{task}/{seed}')
    p.add_argument('--result_path', type=str, default=f'result/{task}/{seed}')
    if os.path.exists(task):
        shutil.rmtree(task)

    torch.backends.cudnn.benchmark = False  # 模型卷积层预先优化关闭


    torch.backends.cudnn.deterministic = True  # 确定为默认卷积算法
    predict_args = p.parse_args()

    predict_args.predict_path =f'dataset/{task}/train.csv'
    model_args = load_args(predict_args.model_path)

    for key, value in vars(predict_args).items():
        setattr(model_args, key, value)
    args = model_args
    if os.path.exists(args.graph_path):
        shutil.rmtree(args.graph_path)
    os.mkdir(args.graph_path)
    args.is_explain = True
    args.use_pyg=True
    setSeed(seed)


    # data = tool.load_data(args)  # all data
    # datas = copy.deepcopy(data)
    #

    test_data = load_data(args,args.predict_path)
    test_data = feature_selected(test_data, args)
    model = load_model(args)
    test_dataset = MixDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    test_pred = predict(model, test_dataloader, args)
    test_pred = np.array(test_pred)
    test_pred = test_pred.tolist()
    target = [x.label for x in test_data]
    smiles = [x.smile for x in test_data]
    test_score = compute_score(target, test_pred, args)
    print(test_score)
    df = pd.DataFrame({'Smiles': smiles, 'target': target})

    df['pred'] = test_pred
    df.to_csv(os.path.join(args.result_path, 'result.csv'))

    # model_args = load_model_args(p.model_path)
    # model_args.model_path = p.model_path


    # random.shuffle(datas)


#     if args.dataset_type == 'classification':
#         train_data, val_data, test_data = (
#             datas[:int(0.8 * len(datas))], datas[int(0.8 * len(datas)):int(0.9 * len(datas))],
#             datas[int(0.9 * len(datas)):] )


#     model = tool.load_model(args)
#     for i in range(len(test_data)):
#         test_data[i].no=i
#     test_dataset = MixDataset(test_data)
#     test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
#     test_pred = tool.predict(model, test_dataloader, model_args, True)
#     test_pred = np.array(test_pred)
#     test_pred = test_pred.tolist()
#     target = [x.label for x in test_data]
#     smiles = [x.smile for x in test_data]
#     no =[x.no for x in test_data]
#     test_score = tool.compute_score(target, test_pred, model_args)
#     df = pd.DataFrame({'no':no,'Smiles': smiles, 'target': target, 'pred': test_pred})
#     df.to_csv(args.predict_path, index=False)
#     print('test auc', test_score)
# path = os.path.join(args.save_path, str(args.seed) , 'model.pt')


if __name__ == '__main__':
    # esol 30 epochs=50, lr=0.003
    #bace 90
    task = 'bace'
    p(task, 90)



