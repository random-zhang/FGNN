import os
import shutil

import torch

from paratmeter import add_train_argument
from train import startrtrain


def compare_weight(args,fp_dim=1400):

    print('start weight')
    print(args.noise_rate)
    print(args.fp_dim)
    avg, std = startrtrain(args)
    print(avg,std)
if __name__=='__main__':
    p = add_train_argument()
    args = p.parse_args()
    args.use_dn = True

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.mode = 'train'
    if os.path.exists(args.graph_path):
        shutil.rmtree(args.graph_path)
    args.task_name = 'tox21'
    if args.task_name == 'bbbp':
        hyperp = {'dropout': 0.4, 'lr': 5e-04, 'gqmnn_hidden_dim1': 300,  # 0.9153247953157081
                  'gqmnn_hidden_dim2': 110, 'gqmnn_hidden_dim3': 32, 'dn_out_dim': 230, 'fpn_hidden_dim': 680,
                  'gat_ci_out': 100,
                  'gat_e_out_dim': 80}
        args.fpn_dropout = 0.6
        args.fpn_out_dim = 600
        args.gnn_dropout = 0.25
        args.nheads = 5
        args.ratio = 0.5
        args.gat_layer1_out_dim = 80
        args.labels = ['Class']
        args.noise_rate = 0.2
        print(args.noise_rate)
        args.weight_decay = 5e-5

        args.fp_dim = 1200
        args.max_atom = 100
        args.batch_size = 50

    elif args.task_name == 'tox21':

        hyperp = {'dropout': 0.4, 'lr': 5e-04, 'gqmnn_hidden_dim1': 300,  # 0.9153247953157081
                  'gqmnn_hidden_dim2': 110, 'gqmnn_hidden_dim3': 32, 'dn_out_dim': 230, 'fpn_hidden_dim': 680,
                  'gat_ci_out': 100,
                  'gat_e_out_dim': 80, 'k_pop': 4}  # 0.9064991885920826
        args.fpn_dropout = 0.05
        args.fpn_out_dim = 400
        args.gnn_dropout = 0.25
        args.nheads = 3
        args.ratio = 0.4
        args.gat_layer1_out_dim = 70
        args.labels = ['SR-MMP']
        args.noise_rate = 0.0
        args.weight_decay = 1e-6  # 9008
        args.noise_rate = 0
    elif args.task_name == 'pdbbind_r':
        hyperp = {'dropout': 0.4, 'lr': 4e-04, 'gqmnn_hidden_dim1': 300,  # 0.9153247953157081
                  'gqmnn_hidden_dim2': 110, 'gqmnn_hidden_dim3': 32, 'dn_out_dim': 230, 'fpn_hidden_dim': 680,
                  'gat_ci_out': 100,
                  'gat_e_out_dim': 80, 'k_pop': 4}
        args.fpn_dropout = 0.3
        args.fpn_out_dim = 350
        args.gnn_dropout = 0.4
        args.nheads = 5
        args.ratio = 0.7
        args.gat_layer1_out_dim = 50
        args.labels = ['Class']
        args.noise_rate = 0.0
        args.weight_decay = 5e-5
        args.dataset_type = 'regression'
        # args.max_atom=90
    elif args.task_name == 'freesolv':
        hyperp = {'dropout': 0.0, 'lr': 6e-04, 'gqmnn_hidden_dim1': 300,  # 0.9153247953157081
                  'gqmnn_hidden_dim2': 110, 'gqmnn_hidden_dim3': 32, 'dn_out_dim': 230, 'fpn_hidden_dim': 680,
                  'gat_ci_out': 100,
                  'gat_e_out_dim': 80, 'k_pop': 4}

        args.fpn_dropout = 0.05
        args.fpn_out_dim = 500
        args.gnn_dropout = 0.1
        args.nheads = 5
        args.ratio = 0.6
        args.gat_layer1_out_dim = 60
        args.labels = ['Class']
        args.noise_rate = 0.0
        args.weight_decay = 5e-5
        args.dataset_type = 'regression'
        args.epochs = 30
    elif args.task_name == 'bace':

        hyperp = {'lr': 5e-04, 'gqmnn_hidden_dim1': 300,  # 0.9153247953157081
                  'gqmnn_hidden_dim2': 110, 'gqmnn_hidden_dim3': 32, 'dn_out_dim': 230, 'fpn_hidden_dim': 680,
                  'gat_ci_out': 100,
                  'gat_e_out_dim': 80}
        args.fpn_dropout = 0.4
        args.fpn_out_dim = 350
        args.gnn_dropout = 0.5
        args.nheads = 3
        args.ratio = 0.7
        args.gat_layer1_out_dim = 60
        args.labels = ['Class']
        args.noise_rate = 0.0

        args.weight_decay = 5e-5

        args.fp_dim = 1200
        args.max_atom = 100
        args.batch_size = 50
        args.dropout = 0.2
        args.fp_dim = 1500

    for key, value in hyperp.items():
        setattr(args, key, value)

    torch.backends.cudnn.benchmark = False  # 模型卷积层预先优化关闭
    torch.backends.cudnn.deterministic = True  # 确定为默认卷积算法
    current_file_path = os.path.abspath(__file__)
    print(current_file_path)
    compare_weight(args)
