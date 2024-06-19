import math
import shutil
from argparse import ArgumentParser, Namespace
import random
from copy import deepcopy

import optuna

import torch
import numpy as np
import os


import tool

import train

import optuna.visualization as vis

def add_argument():
    p = ArgumentParser()

    p.add_argument('--log_path', type=str, default='log',
                   help='The dir of output log file.')
    p.add_argument('--labels', type=list, default=['Class'])
    p.add_argument('--dataset_type', type=str, choices=['classification', 'regression'], default='classification',
                   help='The type of dataset.')
    p.add_argument('--split_ratio', type=float, default=0.9,
                   help='The ratio of data splitting.[train,valid]')
    p.add_argument('--seed', type=int, default=0,  # 3407
                   help='The random seed of model. Using in splitting data.')
    p.add_argument('--epochs', type=int, default=100,
                   help='The number of epochs.')
    p.add_argument('--batch_size', type=int, default=50,
                   help='The size of batch.')
    p.add_argument('--gqmnn_input_dim', type=int, default=1000,
                   help='The dim of hidden layers in model.')
    p.add_argument('--gqmnn_hidden_dim1', type=int, default=400,
                   help='The dim of hidden layers in model.')
    p.add_argument('--gqmnn_hidden_dim2', type=int, default=128,
                   help='The dim of hidden layers in model.')
    p.add_argument('--fp_dim', type=int, default=2513)
    p.add_argument('--fpn_out_dim', type=int, default=300,
                   help='The dim of hidden layers in model.')
    p.add_argument('--fpn_hidden_dim', type=int, default=400,
                   help='The dim of hidden layers in model.')
    p.add_argument('--gat_bonds_input_dim', type=int, default=11,
                   help='The dim of hidden layers in model.')
    p.add_argument('--gat_bonds_out_dim', type=int, default=50,
                   help='The dim of hidden layers in model.')
    p.add_argument('--gat_atom_input_dim', type=int, default=51,
                   help='The dim of hidden layers in model.')
    p.add_argument('--gat_hidden_dim2', type=int, default=480,
                   help='The dim of hidden layers in model.')
    p.add_argument('--gat_e_out_dim', type=int, default=20,
                   help='The dim of hidden layers in model.')
    p.add_argument(
        '--gat_att_in_heads', type=int, default=2,
                   help='The dim of hidden layers in model.')
    p.add_argument('--gat_att_out_dim', type=int, default=60,
                   help='The dim of hidden layers in model.')
    p.add_argument('--gat_out_dim', type=int, default=600,
                   help='The dim of hidden layers in model.')
    p.add_argument('--gat_layer1_out_dim', type=int, default=128,
                   help='The dim of hidden layers in model.')

    p.add_argument('--dn_out_dim', type=int, default=100,
                   help='The dim of hidden layers in model.')

    p.add_argument('--nheads', type=int, default=10,
                   help='The number of the attentions in gnn.')
    p.add_argument('--k_hop', type=int, default=2,
                   help='The number of the attentions in gnn.')
    p.add_argument('--dropout', type=float, default=0.0,
                   help='The dropout of fpn and ffn.')
    p.add_argument('--lr', type=float, default=0.0005,
                   help='The dropout of fpn and ffn.')

    p.add_argument('--task_name', type=str, default='lipo',  # '0.1_all_cluster'
                   help='')
    p.add_argument('--fold_iter', type=int, default=None)
    p.add_argument('--mode', type=str, default='train')
    p.add_argument('--classification_type', type=str, default='ROC')
    p.add_argument('--select_method', type=str, choices=['mi', 'mixed', 'mri', 'jmi', 'cfr', 'ccmi', 'mrmr'],
                   default='selected'
                   )

    p.add_argument('--graph_path', type=str, default='Garph/')
    p.add_argument('--search_num',type=int,default=200)
    p.add_argument('--split_type',choices=['random','scaffold'],default='random')
    p.add_argument('--is_explain', default=False)
    p.add_argument('--use_pyg', default=False)
    return p





def objective(trial):

    args.dropout = trial.suggest_float('dropout', 0.0, 1, step=0.01)
    args.lr = trial.suggest_float('lr', 0.00001, 0.01)






    args.gqmnn_hidden_dim1 = trial.suggest_int('gqmnn_hidden_dim1', 1, 3000)
    args.gqmnn_hidden_dim2 = trial.suggest_int('gqmnn_hidden_dim2', 1, 2000)
    args.gqmnn_hidden_dim3 = trial.suggest_int('gqmnn_hidden_dim3', 1, 1000)

    args.fpn_hidden_dim = trial.suggest_int('fpn_hidden_dim', 1, 2000)
    args.gat_ci_out = trial.suggest_int('gat_ci_out', 1, 500)

    args.gat_e_out_dim = trial.suggest_int('gat_e_out_dim', 1, 500)
    args.weight_decay = trial.suggest_float('weight_decay', 0, 1e-4)
    args.fp_dim=trial.suggest_int('fp_dim',100,2500,step=100)
    
    args.fp_dim=2513

    args.fpn_dropout = trial.suggest_float('fpn_dropout',0,1, step=0.01)
    args.fpn_out_dim = trial.suggest_int('fpn_out_dim', 1, 2000)
    args.gnn_dropout = trial.suggest_float('gnn_dropout', 0, 1, step=0.01)
    args.nheads = trial.suggest_int('nheads', 1, 5)
    args.ratio = trial.suggest_float('ratio', 0.00, 1,  step=0.01)

    args.ratio = 0

    args.gat_layer1_out_dim=trial.suggest_int('gat_layer1_out_dim',1,300)
    if args.dataset_type=='regression':
       args.h_delta=trial.suggest_float('h_delta', 0.01, 1,  step=0.01)
    # if args.task_name=='lipo':
    #     #(0.7283793828797307, 0.0441708787411006, 0.6746531, 0.022525856)
    #     args.fpn_dropout = 0.6
    #     args.fpn_out_dim = 600
    #     args.gnn_dropout = 0.25
    #     args.nheads = 5
    #     args.ratio = 0.5
    #     args.gat_layer1_out_dim = 80

    #print('start:', args.task_name)
    print('serach num:',trial.number)
    metric=train_seed.starttrain(args,deepcopy(data))
    #avg_auc=cross_vaild(args,train_datas,trial.number)# to find the best  hyper-parameter
    print('finish:', args.task_name)
   



    print('val :',metric[2],'test:',metric[0])
    return metric[0]
def setSeed(seed):
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    p = add_argument()
    args = p.parse_args()
    #args.task_name = 'bace'
    if args.task_name=='bace':
        # args.fpn_dropout = 0.4
        # args.fpn_out_dim = 350
        # args.gnn_dropout = 0.5
        # args.nheads = 3
        # args.ratio = 0.7
        # args.gat_layer1_out_dim = 60
        args.labels = ['Class']
    elif args.task_name=='bbbp':
        # args.fpn_dropout = 0.6
        # args.fpn_out_dim = 600
        # args.gnn_dropout = 0.25
        # args.nheads = 5
        # args.ratio = 0.5
        # args.gat_layer1_out_dim = 80
        args.labels=['Class']
        args.dataset_type = 'classification'
    elif args.task_name=='tox21':
        # args.fpn_dropout = 0.05
        # args.fpn_out_dim = 400
        # args.gnn_dropout = 0.25
        # args.nheads = 3...............................00
        # args.ratio = 0.4
        # args.gat_layer1_out_dim = 70
        # args.labels = ['SR-HSE']
        args.dataset_type = 'classification'
    elif args.task_name == 'freesolv':
        # args.fpn_dropout = 0.05
        # args.fpn_out_dim = 500
        # args.gnn_dropout = 0.1
        # args.nheads = 5
        # args.ratio = 0.6
        # args.gat_layer1_out_dim = 60
        args.labels = ['Class']
        args.dataset_type = 'regression'

    elif args.task_name=='pdbbind_c':
        # args.fpn_dropout = 0.1
        # args.fpn_out_dim = 550
        # args.gnn_dropout = 0.15
        # args.nheads = 7
        # args.ratio = 0.6
        # args.gat_layer1_out_dim = 70
        args.labels = ['Class']
        args.dataset_type = 'regression'
    elif args.task_name=='pdbbind_f':
        # args.fpn_dropout = 0.4
        # args.fpn_out_dim = 350
        # args.gnn_dropout = 0.3
        # args.nheads = 2
        # args.ratio = 0.4
        # args.gat_layer1_out_dim = 75
        args.labels = ['Class']
        args.dataset_type = 'regression'
    elif args.task_name=='pdbbind_r':
        # args.fpn_dropout = 0.3
        # args.fpn_out_dim = 350
        # args.gnn_dropout = 0.4
        # args.nheads = 5
        # args.ratio = 0.7
        # args.gat_layer1_out_dim = 50
        args.labels = ['Class']
        args.dataset_type = 'regression'
    elif args.task_name == 'esol':
    #     args.fpn_dropout = 0.05
    #     args.fpn_out_dim = 400
    #     args.gnn_dropout = 0.4
    #     args.nheads = 7
    #     args.ratio = 0.6
    #     args.gat_layer1_out_dim = 65
        args.labels = ['Class']
        args.dataset_type = 'regression'
    elif args.task_name == 'lipo':

        args.labels = ['Class']
        args.dataset_type = 'regression'
    elif args.task_name=='hiv':
        args.labels = ['Class']
        args.dataset_type='classification'
    elif args.task_name=='sider':
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
        args.labels=[args.labels[23]]
        args.dataset_type = 'classification'
    elif args.task_name=='clintox':
        args.labels=['Class1','Class2']
        args.labels=[args.labels[0]]

        args.dataset_type = 'classification'
    elif args.task_name=='muv':
        args.labels=['MUV-466']
        args.dataset_type= 'classification'
        args.classification_type='PRC'
    elif args.task_name=='mda_mb_453' or args.task_name=='mda_mb_231'  or args.task_name=='mda_mb_435'  or args.task_name=='mda_mb_361'  or args.task_name=='mda_mb_468' :
        args.dataset_type = 'classification'
    elif args.task_name=='sk_br_3' or args.task_name=='t_47d' or args.task_name=='mvf_7' or args.task_name=='bt_474' or args.task_name=='bt_20' or  args.task_name=='bt_549':
        args.dataset_type = 'classification'
    elif   args.task_name=='hs_578t' or args.task_name=='bcap37' or args.task_name=='hbl_100':
        args.dataset_type = 'classification'

    args.with_weight = True

    args.with_gci = True

    args.with_rnn = True
    args.with_fc = False

    print('')
    args.noise_rate=0.0
    args.sorted=False
    setSeed(args.seed)
    torch.backends.cudnn.benchmark = False  # 模型卷积层预先优化关闭
    torch.backends.cudnn.deterministic = True  # 确定为默认卷积算法
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.search_now = 0

    db_path='{}.sqlite3'.format(args.task_name)
    if os.path.exists(db_path):
        os.remove(db_path)

    args.data_path = 'dataset/{}/train.csv'.format(args.task_name)
    args.save_path = 'model_save/{}'.format(args.task_name)
    tool.create_dir(args.save_path)
    if  os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    args.max_atom=0
    data = tool.load_data(args)  # all data
    # if len(args.labels) == 1:
    #     data = tool.feature_selected(data, args)
    if args.dataset_type=='classification':
        study = optuna.create_study(direction='maximize', storage='sqlite:///' + db_path)
    else:
        study = optuna.create_study(direction='minimize', storage='sqlite:///' + db_path)
    #
    study.optimize(objective, n_trials=args.search_num)
    vis.plot_optimization_history(study).show()
    vis.plot_intermediate_values(study).show()
    vis.plot_slice(study).show()
    vis.plot_contour(study).show()
    vis.plot_parallel_coordinate(study).show()
    vis.plot_param_importances(study).show()
    # 打印优化结果
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")