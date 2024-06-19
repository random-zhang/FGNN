import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import MI
import tool
from paratmeter import add_train_argument
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


if __name__ == '__main__':
    p = add_train_argument()
    args = p.parse_args()
    #args.use_dn = True

   # args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.mode = 'train'
    # # 设置PyTorch的随机种子
    #
    # torch.backends.cudnn.benchmark = False  # 模型卷积层预先优化关闭
    # torch.backends.cudnn.deterministic = True  # 确定为默认卷积算法
    # tool.setSeed(args.seed)
    args.task_name = 'mcf_7'
    args.data_path = 'dataset/{}/train.csv'.format(args.task_name)

    args.labels = ['Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues',
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
    args.labels = [args.labels[3]]
    args.labels=['Class']
    args.max_atom=0
    data = tool.load_data(args)  # all data
    #
    # if args.dataset_type == 'classification':
    #     y = np.asarray([int(d.label) for d in data])
    #     train_datas, test_data, _, _ = train_test_split(data, y, test_size=0.1, stratify=y, random_state=args.seed)
    #     # train_datas, test_data = train_test_split(data, test_size=0.1, random_state=args.seed)
    # else:
    #     train_datas, test_data = train_test_split(data, test_size=0.1)
    fps = []
    labels=[]
    for one in data:
        fps.append(one.fp)
        labels.append(one.label[0])
    fps = np.asarray(fps)
    labels=np.asarray(labels)
    df=pd.DataFrame(fps)
    df['target']=labels
    df.to_csv(f'{args.task_name}_{args.labels[0]}_fps.csv',index=False)


    #start(args.task_name,args.labels,df)





