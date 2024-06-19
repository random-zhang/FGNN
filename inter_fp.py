import copy
import os
from argparse import ArgumentParser
import shap
import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import pandas as pd

from Data.Dataset import MixDataset
from tool import *
from train import load_model_args


def pp(args,tmp):
    model = load_model(args)
    fps=[d.fp for d in tmp]
    fps=torch.tensor(fps).float().cuda()
    # pred=model.encoder2(fps)
    # explainer = shap.DeepExplainer(model.encoder2,fps)
    # shap_values = explainer.shap_values(fps)
    # fig, ax = plt.subplots(figsize=(10,10))
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = 'Times New Roman'
    # shap.summary_plot(shap_values, fps, show=False, max_display=10, class_names=None)  # , plot_type="bar"
    # plt.xticks(fontproperties='Times New Roman', size=15)  # 设置x坐标字体和大小
    # plt.yticks(fontproperties='Times New Roman', size=15)  # 设置y坐标字体和大小
    # plt.tick_params(axis='both', direction='in',length=5,width=2)
    # plt.xlabel('Mean(average impact on model output magnitude)', fontsize=20)  # 设置x轴标签和大小
    # plt.tight_layout()
    # plt.legend(bbox_to_anchor=(1.5, 1.5))
    # # 让坐标充分显示，如果没有这一行，坐标可能显示不全
    # plt.savefig("保存.png", dpi=300)  # 可以保存图片

    #shap.plot_explanation(shap_values, fps, show=True, max_display=10, class_names=[], plot_type="bar")



    test_dataset = MixDataset(tmp)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    test_pred = predict(model, test_dataloader, args)
    test_pred = np.array(test_pred)
    test_pred = test_pred.tolist()
    target = [x.label for x in tmp]

    test_score = compute_score(target, test_pred, args)
    return test_score





if __name__ == '__main__':

    task = 'mda_mb_361'



    torch.backends.cudnn.benchmark = False  # 模型卷积层预先优化关闭

    torch.backends.cudnn.deterministic = True  # 确定为默认卷积算法

    list=[]
    for idx in range(700):
       aucs = []
       for seed in range(10,110,10):
       #for seed in [60]:
          setSeed(seed)
          p = ArgumentParser()
          p.add_argument('--model_path', type=str, default=f'model_save/{task}/{seed}/model.pt')
          p.add_argument('--predict_path', type=str, default=f'model_save/{task}/{seed}/predict.csv')
          p.add_argument('--graph_path', type=str, default=f'result/{task}/{seed}')
          p.add_argument('--result_path', type=str, default=f'result/{task}/{seed}')
          predict_args = p.parse_args()
          model_args = load_args(predict_args.model_path)

          for key, value in vars(predict_args).items():
              setattr(model_args, key, value)
          args = model_args

          test_data = load_data(args, args.predict_path)
          test_data = feature_selected(test_data, args)
          tmp = copy.deepcopy(test_data)

          for i in range(len(tmp)):

              tmp[i].fp[idx] = 1 - tmp[i].fp[idx]

          aucs.append(pp(args, tmp))
       aucs=np.array(aucs)
       list.append(aucs.mean())
       print(idx,aucs.mean())
    print(list)


    # seed=90
    # p = ArgumentParser()
    # p.add_argument('--model_path', type=str, default=f'model_save/{task}/{seed}/model.pt')
    # p.add_argument('--predict_path', type=str, default=f'model_save/{task}/{seed}/predict.csv')
    # p.add_argument('--graph_path', type=str, default=f'result/{task}/{seed}')
    # p.add_argument('--result_path', type=str, default=f'result/{task}/{seed}')
    # predict_args = p.parse_args()
    # model_args = load_args(predict_args.model_path)
    #
    # for key, value in vars(predict_args).items():
    #     setattr(model_args, key, value)
    # args = model_args
    #
    # setSeed(seed)
    # test_data = load_data(args, args.predict_path)
    # #test_data = feature_selected(test_data, args)
    #
    # tmp = copy.deepcopy(test_data)
    # aucs=pp(args,tmp)
    # print(aucs )




