import os
import shutil
from copy import deepcopy

import torch

import tool
import train
from paratmeter import add_train_argument, set_hyper_parameter



def fp_seletor(args):
    print('start selector')
    args.sorted = False
    args.data_path = 'dataset/{}/train.csv'.format(args.task_name)
    raw_data=tool.load_data(args)

    #1400    10 0.8651025956219563  20  0.8529280037156926
    metric = []
    for i in range(100, 2513, 100):
        args.fp_dim = i
        print(args.fp_dim)

        result = train_seed.starttrain(args,deepcopy(raw_data))
        metric.append(result)
        print(result)
    print(metric)
    # metric = []
    # args.sorted = True
    # for i in range(100, 2513, 100):
    #     args.fp_dim = i
    #     print(args.fp_dim)
    #     result = starttrain(args)
    #     metric.append(result)
    #     print(result)

    # print(metric)

if __name__=='__main__':
    p = add_train_argument()
    args = p.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.mode = 'train'
    if os.path.exists(args.graph_path):
        shutil.rmtree(args.graph_path)

    torch.backends.cudnn.benchmark = False  # 模型卷积层预先优化关闭

    torch.backends.cudnn.deterministic = True  # 确定为默认卷积算法
    args = set_hyper_parameter(args)
    stds = []
    avgs = []

    # args.select_method = 'mixed'
    args.with_weight = True
    args.with_gci = True
    args.with_rnn = True
    args.with_fc = False
    args.select_method='mim'
    print(args.fp_dim)
    #train_seed.fp_2513(args)
    print('noise', args.noise_rate)
    print('select_mode', args.select_method)
    print('split type:', args.split_type)


    fp_seletor(args)


