from argparse import ArgumentParser


def add_train_argument():
    p = ArgumentParser()

    p.add_argument('--log_path', type=str, default='log',
                   help='The dir of output log file.')
    p.add_argument('--labels',type=list,default=['Class'])
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
    p.add_argument('--fp_dim',type=int,default=2513)
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
    p.add_argument('--gat_att_in_heads', type=int, default=2,
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
    p.add_argument('--leakyrelu_aplha', type=float, default=0.2,
                   help='The alpha of leakyrelu .')
    p.add_argument('--task_name', type=str, default='hiv',  # '0.1_all_cluster'
                   help='')
    p.add_argument('--fold_iter', type=int, default=None)
    p.add_argument('--mode', type=str, default='train')
    p.add_argument('--classification_type', type=str, default='ROC')
    p.add_argument('--graph_path',type=str,default='Garph/')
    p.add_argument('--gci_graph_path', type=str, default='Garph/GCI/')
    p.add_argument('--select_method',type=str,choices=['mi','mixed','mri','jmi','cfr','ccmi','mrmr','cife','mim'],default='selected'
                                                                                                             )
    p.add_argument('--with_weight',type=bool,choices=[True,False],default=True)
    p.add_argument('--noise_rate',type=float,default=0.0)
    p.add_argument('--split_type',choices=['random','scaffold'],default='random')
    p.add_argument('--test_path',type=str)
    p.add_argument('--ratio',type=float,default=0)

    p.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    p.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    p.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    p.add_argument('--limit_num',type=int,default=-1)
    p.add_argument('--is_explain',default=False)
    p.add_argument('--use_pyg',default=False)
    p.add_argument('--with_gci',default=True)

    return p

def set_hyper_parameter(args):
    if args.task_name=='bbbp' :#(0.9500151751185009, 0.015502028605851896, 0.953527524730827, 0.015058844456749492)
        if args.split_type=='random':
            hyperp = {'dropout': 0.3, 'lr': 0.001, 'gqmnn_hidden_dim1': 300,
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
            args.noise_rate = 0.0
            args.weight_decay =1e-6
            args.fp_dim=800
        else:
            # (0.9336505317005506, 0.030283171894775307, 0.9426718024985767, 0.04016830989946874)
            #700 (0.9522932712924238, 0.01769437255707673, 0.9602448944982953, 0.017696934018513216)
            hyperp = {'dropout': 0.4, 'lr': 0.001, 'gqmnn_hidden_dim1': 300,
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
            args.noise_rate = 0.0
            args.dataset_type = 'classification'
            args.weight_decay = 1e-6
    elif args.task_name=='tox21' :
        if  args.labels == ['NR-AR']:
            #(0.8441998911289641, 0.0566993438615395, 0.8475918351994498, 0.0329846009902272) fpn 400
            #(0.8328268932274103, 0.03851178157105218, 0.8489966172077246, 0.036616007030212706)fgmnn 400
            #(0.7351728116578218, 0.06307649168586657, 0.7528215783140683, 0.061724619969568066)fpn 2513
            #(0.7308403753455213, 0.06690332509051848, 0.654061915008565, 0.1374777559393858)fgmnn 2513
            #(0.7278185289378017, 0.09712976414739191, 0.7235005267565414, 0.03773605654358422) gnn
            hyperp = {'dropout': 0.61, 'lr': 0.002461811246109409, 'gqmnn_hidden_dim1': 2524, 'gqmnn_hidden_dim2': 1769,
                      'gqmnn_hidden_dim3': 845, 'fpn_hidden_dim': 1548, 'gat_ci_out': 93, 'gat_e_out_dim': 354,
                      'weight_decay': 4.891687839748576e-05, 'fp_dim': 400, 'fpn_dropout': 0.54, 'fpn_out_dim': 1699,
                      'gnn_dropout': 0.52, 'nheads': 4, 'gat_layer1_out_dim': 212}
            hyperp['ratio'] = 1
            # hyperp = {'dropout': 0.61, 'lr': 0.002461811246109409, 'gqmnn_hidden_dim1': 2524, 'gqmnn_hidden_dim2': 1769,
            #           'gqmnn_hidden_dim3': 845, 'fpn_hidden_dim': 1548, 'gat_ci_out': 93, 'gat_e_out_dim': 300,
            #           'weight_decay': 4.891687839748576e-05, 'fp_dim': 400, 'fpn_dropout': 0.54, 'fpn_out_dim': 1699,
            #           'gnn_dropout': 0.52, 'nheads': 3, 'gat_layer1_out_dim': 212}
            # hyperp['ratio'] = 0.15
        elif args.labels == ['NR-AR-LBD']:
            #(0.9113752920264588, 0.034433102697059105, 0.9127365975499477, 0.03264605057290402) 400 all
            hyperp = {'dropout': 0.4, 'lr': 5e-04, 'gqmnn_hidden_dim1': 300,  # 0.9153247953157081
                      'gqmnn_hidden_dim2': 110, 'gqmnn_hidden_dim3': 32, 'dn_out_dim': 230, 'fpn_hidden_dim': 680,
                      'gat_ci_out': 100,
                      'gat_e_out_dim': 80}  # 0.9064991885920826
            args.fpn_dropout = 0.05
            args.fpn_out_dim = 400
            args.gnn_dropout = 0.25
            args.nheads = 3
            args.ratio = 0.4
            args.gat_layer1_out_dim = 70#as70
            args.weight_decay = 1e-6

        elif args.labels == ['NR-AhR']:

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

            args.noise_rate = 0.0
            args.weight_decay = 1e-6
            args.noise_rate = 0
            #(0.881109965847369, 0.016834969475624434, 0.9117593919487643, 0.020786162223709003)
            hyperp={'dropout': 0.0, 'lr': 0.00433, 'gqmnn_hidden_dim1': 336, 'gqmnn_hidden_dim2': 671, 'gqmnn_hidden_dim3': 82, 'fpn_hidden_dim': 801, 'gat_ci_out': 405, 'gat_e_out_dim': 320, 'weight_decay': 2.1e-05, 'fpn_dropout': 0.6, 'fpn_out_dim': 324, 'gnn_dropout': 0.1, 'nheads': 3, 'ratio': 0.3, 'gat_layer1_out_dim': 36}
            hyperp={'dropout': 0.1, 'lr': 0.0017, 'gqmnn_hidden_dim1': 12, 'gqmnn_hidden_dim2': 236, 'gqmnn_hidden_dim3': 153, 'fpn_hidden_dim': 775, 'gat_ci_out': 471, 'gat_e_out_dim': 405, 'weight_decay': 0.00158, 'fpn_dropout': 0.7, 'fpn_out_dim': 812, 'gnn_dropout': 0.2, 'nheads': 1, 'ratio': 0.5, 'gat_layer1_out_dim': 40}
            hyperp={'dropout': 0.2, 'lr': 0.0019, 'gqmnn_hidden_dim1': 123, 'gqmnn_hidden_dim2': 293, 'gqmnn_hidden_dim3': 204, 'fpn_hidden_dim': 288, 'gat_ci_out': 248, 'gat_e_out_dim': 373, 'weight_decay': 0.0017, 'fpn_dropout': 0.6, 'fpn_out_dim': 943, 'gnn_dropout': 0.2, 'nheads': 1, 'ratio': 0.8, 'gat_layer1_out_dim': 53}
           #(0.8946049522077152, 0.0170032236843397, 0.9251862427532339, 0.017856359776329506)
            hyperp = {'dropout': 0.2, 'lr': 0.00159, 'gqmnn_hidden_dim1': 186, 'gqmnn_hidden_dim2': 258,
                      'gqmnn_hidden_dim3': 196, 'fpn_hidden_dim': 232, 'gat_ci_out': 163, 'gat_e_out_dim': 328,
                      'weight_decay': 0.00095, 'fpn_dropout': 0.7, 'fpn_out_dim': 804, 'gnn_dropout': 0.2, 'nheads': 1,
                      'ratio': 0.9, 'gat_layer1_out_dim': 80}
            #Trial 69 finished with value: 0.8860145809444392 and parameters: {'dropout': 0.2, 'lr': 0.001585610462212104, 'gqmnn_hidden_dim1': 186, 'gqmnn_hidden_dim2': 258, 'gqmnn_hidden_dim3': 196, 'fpn_hidden_dim': 232, 'gat_ci_out': 163, 'gat_e_out_dim': 328, 'weight_decay': 0.0009450476715590302, 'fpn_dropout': 0.7000000000000001, 'fpn_out_dim': 804, 'gnn_dropout': 0.2, 'nheads': 1, 'ratio': 0.9, 'gat_layer1_out_dim': 80}. Best is trial 69 with value: 0.8860145809444392.
        elif args.labels == ['NR-Aromatase']:
            #(0.8419571875837446, 0.029880683893175674, 0.8593952937806743, 0.030430773655687197) 1000

            hyperp = {'dropout': 0.6, 'lr': 0.00011, 'gqmnn_hidden_dim1': 62, 'gqmnn_hidden_dim2': 960,
                      'gqmnn_hidden_dim3': 229, 'fpn_hidden_dim': 327, 'gat_ci_out': 380, 'gat_e_out_dim': 237,
                      'weight_decay': 0.0076, 'fpn_dropout': 0.8, 'fpn_out_dim': 706, 'gnn_dropout': 0.2, 'nheads': 4,
                      'ratio': 0.3, 'gat_layer1_out_dim': 194}
            hyperp = {'dropout': 0.6, 'lr': 0.00011, 'gqmnn_hidden_dim1': 62, 'gqmnn_hidden_dim2': 960,
                      'gqmnn_hidden_dim3': 229, 'fpn_hidden_dim': 327, 'gat_ci_out': 380, 'gat_e_out_dim': 237,
                      'weight_decay': 0.0076, 'fpn_dropout': 0.8, 'fpn_out_dim': 706, 'gnn_dropout': 0.2, 'nheads': 4,
                      'ratio': 0.3, 'gat_layer1_out_dim': 194}
        elif args.labels == ['NR-ER']:
            # (0.7538128313188588, 0.026601267357434042, 0.7700767265366174, 0.03616819416767344) 400
            hyperp = {'dropout': 0.4, 'lr': 5e-04, 'gqmnn_hidden_dim1': 300,
                      'gqmnn_hidden_dim2': 110, 'gqmnn_hidden_dim3': 32, 'dn_out_dim': 230, 'fpn_hidden_dim': 680,
                      'gat_ci_out': 100,
                      'gat_e_out_dim': 80, 'k_pop': 4}  # 0.9064991885920826
            args.fpn_dropout = 0.05
            args.fpn_out_dim = 400
            args.gnn_dropout = 0.25
            args.nheads = 3
            args.ratio = 0.4
            args.gat_layer1_out_dim = 70

            args.noise_rate = 0.0
            args.weight_decay = 1e-6
            # (0.736438220285485, 0.02569132856913232, 0.760733939698847, 0.025336832313358083)
            hyperp = {'dropout': 0.8, 'lr': 0.000388, 'gqmnn_hidden_dim1': 119, 'gqmnn_hidden_dim2': 451,
                      'gqmnn_hidden_dim3': 274, 'fpn_hidden_dim': 689, 'gat_ci_out': 451, 'gat_e_out_dim': 34,
                      'weight_decay': 0.00266, 'fpn_dropout': 0.6, 'fpn_out_dim': 292, 'gnn_dropout': 0.5, 'nheads': 2,
                      'ratio': 0.4, 'gat_layer1_out_dim': 119}
            #(0.7391758893348703, 0.015521205037687887, 0.764138228425581, 0.024153860621972768)
            hyperp = {'dropout': 0.5, 'lr': 0.000388, 'gqmnn_hidden_dim1': 119, 'gqmnn_hidden_dim2': 451,
                      'gqmnn_hidden_dim3': 274, 'fpn_hidden_dim': 689, 'gat_ci_out': 451, 'gat_e_out_dim': 34,
                      'weight_decay': 0.00266, 'fpn_dropout': 0.6, 'fpn_out_dim': 292, 'gnn_dropout': 0.5, 'nheads': 2,
                      'ratio': 0.4, 'gat_layer1_out_dim': 119}
            #(0.7299421213614457, 0.022026740294519287, 0.7602567382956951, 0.023186344622523045)
            hyperp = {'dropout': 0.0, 'lr': 0.000388, 'gqmnn_hidden_dim1': 119, 'gqmnn_hidden_dim2': 451,
                      'gqmnn_hidden_dim3': 274, 'fpn_hidden_dim': 689, 'gat_ci_out': 451, 'gat_e_out_dim': 34,
                      'weight_decay': 0.00266, 'fpn_dropout': 0.6, 'fpn_out_dim': 292, 'gnn_dropout': 0.5, 'nheads': 2,
                      'ratio': 0.4, 'gat_layer1_out_dim': 119}
            #(0.7391269976272643, 0.01977863353199289, 0.7629445004935059, 0.022761501876442426)
            hyperp = {'dropout': 0.6, 'lr': 0.000388, 'gqmnn_hidden_dim1': 119, 'gqmnn_hidden_dim2': 451,
                      'gqmnn_hidden_dim3': 274, 'fpn_hidden_dim': 689, 'gat_ci_out': 451, 'gat_e_out_dim': 34,
                      'weight_decay': 0.00266, 'fpn_dropout': 0.6, 'fpn_out_dim': 292, 'gnn_dropout': 0.5, 'nheads': 2,
                      'ratio': 0.4, 'gat_layer1_out_dim': 119}
            #(0.7359989308120014, 0.020191662750963933, 0.7614317980636545, 0.02300059073500683)
            hyperp = {'dropout': 0.4, 'lr': 0.000388, 'gqmnn_hidden_dim1': 119, 'gqmnn_hidden_dim2': 451,
                      'gqmnn_hidden_dim3': 274, 'fpn_hidden_dim': 689, 'gat_ci_out': 451, 'gat_e_out_dim': 34,
                      'weight_decay': 0.00266, 'fpn_dropout': 0.6, 'fpn_out_dim': 292, 'gnn_dropout': 0.5, 'nheads': 2,
                      'ratio': 0.4, 'gat_layer1_out_dim': 119}
            hyperp = {'dropout': 0.5, 'lr': 0.000388, 'gqmnn_hidden_dim1': 119, 'gqmnn_hidden_dim2': 451,
                      'gqmnn_hidden_dim3': 274, 'fpn_hidden_dim': 689, 'gat_ci_out': 451, 'gat_e_out_dim': 34,
                      'weight_decay': 0.00266, 'fpn_dropout': 0.6, 'fpn_out_dim': 292, 'gnn_dropout': 0.5, 'nheads': 2,
                      'ratio': 0.4, 'gat_layer1_out_dim': 119}

            # (0.757396599685456, 0.026878145093290236, 0.7647109154894075, 0.04352946942040635) 400 all
            # (0.7676925448109174, 0.02249659397549852, 0.7770628394419522, 0.03859239952861163) 500 all
            # (0.7649774289162372, 0.022262406220306875, 0.779391309852309, 0.03806208939905575)600
            #(0.7645436868722699, 0.02203604484764082, 0.7797788649339388, 0.037222100950727456)700 all
            # (0.7562787815600747, 0.019267256276485168, 0.7704068851516908, 0.03762853669632737)800
            #(0.7535798489582096, 0.02093618272792793, 0.7637704004816208, 0.03198558516561568)900
            #(0.7380619514968736, 0.028448389283670033, 0.7641492622495201, 0.03467128433526174) all
            hyperp = {'dropout': 0.5, 'lr': 0.000388, 'gqmnn_hidden_dim1': 119, 'gqmnn_hidden_dim2': 451,
                      'gqmnn_hidden_dim3': 274, 'fpn_hidden_dim': 689, 'gat_ci_out': 451, 'gat_e_out_dim': 34,
                      'weight_decay': 0.00266, 'fpn_dropout': 0.6, 'fpn_out_dim': 292, 'gnn_dropout': 0.5, 'nheads': 2,
                      'ratio': 0.4, 'gat_layer1_out_dim': 119}
        elif args.labels == ['NR-ER-LBD']:
            #(0.8086652532627434, 0.041852825358537245, 0.8277543499411484, 0.042262353409420805) 100 all
            #(0.8192517204928207, 0.04052774135646855, 0.8418507907117796, 0.0660204156178174) 200 all
            #(0.8314135561539711, 0.045134212443460546, 0.8523486002007552, 0.05885179005682355)300 all
            #(0.8308790774265585, 0.04229859243923336, 0.8616347121433506, 0.05037220826358183) 400 all
            #(0.817324755510921, 0.03581603434825925, 0.8479724586204768, 0.04490055570782557) 500 all
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

            args.noise_rate = 0.0
            args.weight_decay = 1e-6
            args.noise_rate = 0
            # (0.8427824867914738, 0.036966996940589456, 0.8728855426932437, 0.04588290367307952) 400
            hyperp = {'dropout': 0.4, 'lr': 5e-04, 'gqmnn_hidden_dim1': 300,  # 0.9153247953157081
                      'gqmnn_hidden_dim2': 110, 'gqmnn_hidden_dim3': 32, 'dn_out_dim': 230, 'fpn_hidden_dim': 680,
                      'gat_ci_out': 100,
                      'gat_e_out_dim': 80}  # 0.9064991885920826
            args.fpn_dropout = 0.4
            args.fpn_out_dim = 400
            args.gnn_dropout = 0.25
            args.nheads = 3
            args.ratio = 0.4
            args.gat_layer1_out_dim = 70
            args.weight_decay = 1e-6
            #(0.8464221834555932, 0.032815214887325625, 0.8800662698965871, 0.04157668949489707)
            hyperp = {'dropout': 0.4, 'lr': 5e-04, 'gqmnn_hidden_dim1': 300,  # 0.9153247953157081
                      'gqmnn_hidden_dim2': 110, 'gqmnn_hidden_dim3': 32, 'dn_out_dim': 230, 'fpn_hidden_dim': 680,
                      'gat_ci_out': 100,
                      'gat_e_out_dim': 80}  # 0.9064991885920826
            args.fpn_dropout = 0.4
            args.fpn_out_dim = 400
            args.gnn_dropout = 0.25
            args.nheads = 3
            args.ratio = 0.4
            args.gat_layer1_out_dim = 70
            args.weight_decay = 1e-3

        elif args.labels == ['NR-PPAR-gamma']:

            #(0.8591665059170465, 0.05019427719063436, 0.8753376154629464, 0.03511294296047563),400
            hyperp={'dropout': 0.8, 'lr': 0.000388, 'gqmnn_hidden_dim1': 119, 'gqmnn_hidden_dim2': 451, 'gqmnn_hidden_dim3': 274, 'fpn_hidden_dim': 689, 'gat_ci_out': 451, 'gat_e_out_dim': 34, 'weight_decay': 0.00266, 'fpn_dropout': 0.6, 'fpn_out_dim': 292, 'gnn_dropout': 0.5, 'nheads': 2, 'ratio': 0.4, 'gat_layer1_out_dim': 119}
            # (0.8615621885817083, 0.04651497112367318, 0.8963338795974207, 0.034029936117712975) 400
            hyperp = {'dropout': 0.9, 'lr': 0.000388, 'gqmnn_hidden_dim1': 119, 'gqmnn_hidden_dim2': 451,
                      'gqmnn_hidden_dim3': 274, 'fpn_hidden_dim': 689, 'gat_ci_out': 451, 'gat_e_out_dim': 34,
                      'weight_decay': 0.00266, 'fpn_dropout': 0.6, 'fpn_out_dim': 292, 'gnn_dropout': 0.5, 'nheads': 2,
                      'ratio': 0.4, 'gat_layer1_out_dim': 119}
        elif args.labels == ['SR-ARE']:
            #(0.8124974885903317, 0.02491708931602263, 0.806618354923707, 0.02005421209735801)
            #(0.8242615867026799, 0.023024047185195275, 0.8223890269803773, 0.01805735906062475) 300
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

            args.noise_rate = 0.0
            args.weight_decay = 1e-6
            args.noise_rate = 0
            #(0.8324472332044959, 0.023506135682905, 0.8306317730140915, 0.02503916445944534)
            #(0.8401736154664412, 0.022072988204407815, 0.8329121668989172, 0.024795661765753204) 600
            hyperp={'dropout': 0.4, 'lr': 0.001124, 'gqmnn_hidden_dim1': 195, 'gqmnn_hidden_dim2': 911,
             'gqmnn_hidden_dim3': 11, 'fpn_hidden_dim': 779, 'gat_ci_out': 79, 'gat_e_out_dim': 187,
             'weight_decay': 0.00119, 'fpn_dropout': 0.5, 'fpn_out_dim': 746, 'gnn_dropout': 0.2,
             'nheads': 2, 'ratio': 0.3, 'gat_layer1_out_dim': 109}
            #(0.8418386771749319, 0.019352781445187955, 0.8339591192767779, 0.026527839677081406)
            #(0.8418386771749319, 0.019352781445187955, 0.8339591192767779, 0.026527839677081406) 600
            hyperp = {'dropout': 0.3, 'lr': 0.001124, 'gqmnn_hidden_dim1': 195, 'gqmnn_hidden_dim2': 911,
                      'gqmnn_hidden_dim3': 11, 'fpn_hidden_dim': 779, 'gat_ci_out': 79, 'gat_e_out_dim': 187,
                      'weight_decay': 0.00119, 'fpn_dropout': 0.5, 'fpn_out_dim': 746, 'gnn_dropout': 0.2,
                      'nheads': 2, 'ratio': 0.3, 'gat_layer1_out_dim': 109}


        elif args.labels == ['SR-ATAD5']:
           #(0.8825121774389262, 0.04056844679645745, 0.8863349454924123, 0.039198181473701695) 500
            hyperp = {'dropout': 0.8, 'lr': 0.000388, 'gqmnn_hidden_dim1': 119, 'gqmnn_hidden_dim2': 451,
                      'gqmnn_hidden_dim3': 274, 'fpn_hidden_dim': 689, 'gat_ci_out': 451, 'gat_e_out_dim': 34,
                      'weight_decay': 0.00266, 'fpn_dropout': 0.6, 'fpn_out_dim': 292, 'gnn_dropout': 0.5, 'nheads': 2,
                      'ratio': 0.4, 'gat_layer1_out_dim': 119}
        elif args.labels == ['SR-HSE']:
            # (0.7694763037261378, 0.038218090347073394, 0.735500641675636, 0.058945087965981184)
            hyperp = {'dropout': 0.3, 'lr': 0.0003278, 'gqmnn_hidden_dim1': 140, 'gqmnn_hidden_dim2': 828,
                      'gqmnn_hidden_dim3': 117, 'fpn_hidden_dim': 960, 'gat_ci_out': 497, 'gat_e_out_dim': 281,
                      'weight_decay': 0.007175, 'fpn_dropout': 0.5, 'fpn_out_dim': 750, 'gnn_dropout': 1.0, 'nheads': 4,
                      'ratio': 0.1, 'gat_layer1_out_dim': 11}
            # {'dropout': 0.6000000000000001, 'lr': 0.0008709866643804332,
            #                                         'gqmnn_hidden_dim1': 227, 'gqmnn_hidden_dim2': 585,
            #                                         'gqmnn_hidden_dim3': 171, 'fpn_hidden_dim': 750, 'gat_ci_out': 228,
            #                                         'gat_e_out_dim': 249, 'weight_decay': 0.009462745541417256,
            #                                         'fpn_dropout': 0.4, 'fpn_out_dim': 339, 'gnn_dropout': 0.4,
            #                                         'nheads': 2, 'ratio': 0.1, 'gat_layer1_out_dim': 239}
            #600(0.8124913879193549, 0.03995915759467077, 0.8014134067174741, 0.03160027026655812) 600
            #(0.8162150898197057, 0.04703257370138099, 0.8020953719722355, 0.03520144718987475) 500
            hyperp={'dropout': 0.6, 'lr': 0.00087,
             'gqmnn_hidden_dim1': 227, 'gqmnn_hidden_dim2': 585,
             'gqmnn_hidden_dim3': 171, 'fpn_hidden_dim': 750, 'gat_ci_out': 228,
             'gat_e_out_dim': 249, 'weight_decay': 0.00946,
             'fpn_dropout': 0.4, 'fpn_out_dim': 339, 'gnn_dropout': 0.4,
             'nheads': 2, 'ratio': 0.1, 'gat_layer1_out_dim': 239}

            #[I 2023-11-06 11:44:50,634] Trial 4 finished with value: 0.7708294898571711 and parameters: {'dropout': 0.7000000000000001, 'lr': 0.0002946731509978517, 'gqmnn_hidden_dim1': 386, 'gqmnn_hidden_dim2': 798, 'gqmnn_hidden_dim3': 103, 'fpn_hidden_dim': 457, 'gat_ci_out': 308, 'gat_e_out_dim': 309, 'weight_decay': 0.007241700460320018, 'fpn_dropout': 0.30000000000000004, 'fpn_out_dim': 724, 'gnn_dropout': 0.9, 'nheads': 1, 'ratio': 0.9, 'gat_layer1_out_dim': 180}. Best is trial 4 with value: 0.7708294898571711.

        elif   args.labels == ['SR-MMP']:

            #(0.9212733502253755, 0.01208874727582395, 0.9310367534500082, 0.010342533421451552), 2400
            hyperp = {'dropout': 0.3, 'lr': 0.0008656, 'gqmnn_hidden_dim1': 516, 'gqmnn_hidden_dim2': 255,
                      'gqmnn_hidden_dim3': 132, 'fpn_hidden_dim': 855, 'gat_ci_out': 215, 'gat_e_out_dim': 311,
                      'weight_decay': 0.0031, 'fpn_dropout': 0.6, 'fpn_out_dim': 304, 'gnn_dropout': 0.5, 'nheads': 1,
                      'ratio': 0.3, 'gat_layer1_out_dim': 126}

        elif args.labels == ['SR-p53']:
            #(0.8609504055144134, 0.03667116340505017, 0.865808403632417, 0.0237221889454374)
            hyperp = {'dropout': 0.4, 'lr': 0.0001, 'gqmnn_hidden_dim1': 748, 'gqmnn_hidden_dim2': 187,
                      'gqmnn_hidden_dim3': 174, 'fpn_hidden_dim': 524, 'gat_ci_out': 140, 'gat_e_out_dim': 450,
                      'weight_decay': 0.008, 'fpn_dropout': 0.4, 'fpn_out_dim': 675, 'gnn_dropout': 0.6, 'nheads': 3,
                      'ratio': 0.3, 'gat_layer1_out_dim': 221}
            hyperp = {'dropout': 0.4, 'lr': 0.0001, 'gqmnn_hidden_dim1': 748, 'gqmnn_hidden_dim2': 187,
                  'gqmnn_hidden_dim3': 174, 'fpn_hidden_dim': 524, 'gat_ci_out': 140, 'gat_e_out_dim': 450,
                  'weight_decay': 0.008, 'fpn_dropout': 0.6, 'fpn_out_dim': 675, 'gnn_dropout': 0.6, 'nheads': 4,
                  'ratio': 0.3, 'gat_layer1_out_dim': 221}
            hyperp = {'dropout': 0.21, 'lr': 2.3450260643449973e-05, 'gqmnn_hidden_dim1': 2899,
                      'gqmnn_hidden_dim2': 788, 'gqmnn_hidden_dim3': 179, 'fpn_hidden_dim': 1069, 'gat_ci_out': 144,
                      'gat_e_out_dim': 114, 'weight_decay': 6.337853705550375e-05, 'fp_dim': 400, 'fpn_dropout': 0.34,
                      'fpn_out_dim': 1825, 'gnn_dropout': 0.4, 'nheads': 5, 'gat_layer1_out_dim': 100,'ratio':0}

           #0.8213825501606362.
        # {'dropout': 0.0, 'lr': 0.0006975125226712842, 'gqmnn_hidden_dim1': 981, 'gqmnn_hidden_dim2': 283,
        #  'gqmnn_hidden_dim3': 85, 'fpn_hidden_dim': 593, 'gat_ci_out': 241, 'gat_e_out_dim': 390,
        #  'weight_decay': 0.0049994709589711315, 'fpn_dropout': 0.6000000000000001, 'fpn_out_dim': 121,
        #  'gnn_dropout': 0.30000000000000004, 'nheads': 4, 'ratio': 0.1, 'gat_layer1_out_dim': 128}

        else:
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

            args.noise_rate = 0.0
            args.weight_decay = 1e-6
            args.noise_rate = 0
            args.noise_rate = 0


    elif args.task_name=='pdbbind_r':
        # hyperp={'dropout':0.1,'lr':0.004285128901327543,'gqmnn_hidden_dim1':772,'gqmnn_hidden_dim2':146,'gqmnn_hidden_dim3':178,'fpn_hidden_dim':781,
        #         'gat_ci_out': 135,'gat_e_out_dim':70,'weight_decay':0.008686006687883184, 'fpn_dropout':0.7000000000000001,
        #         'fpn_out_dim': 685,'gnn_dropout':0.6000000000000001,'nheads':1,'ratio':0.30000000000000004,'gat_layer1_out_dim':47}
        # (1.3350592425210324, 0.03619136832276581, 1.3116307, 0.026218416)
        hyperp = {'dropout': 0.1, 'lr': 0.003, 'gqmnn_hidden_dim1': 772, 'gqmnn_hidden_dim2': 146,
                  'gqmnn_hidden_dim3': 178, 'fpn_hidden_dim': 781,
                  'gat_ci_out': 135, 'gat_e_out_dim': 70, 'weight_decay': 0.009,
                  'fpn_dropout': 0.7,
                  'fpn_out_dim': 685, 'gnn_dropout': 0.6, 'nheads': 1, 'ratio': 0.3,
                  'gat_layer1_out_dim': 47,'h_delta':1}


        args.labels = ['Class']
        args.dataset_type = 'regression'
    elif args.task_name=='freesolv':
        args.labels = ['Class']
        args.dataset_type = 'regression'

        # (0.9536158188749477, 0.1638076961482685, 0.98618287, 0.14482193)
        hyperp = {'dropout': 0.06, 'lr': 0.000628, 'gqmnn_hidden_dim1': 211, 'gqmnn_hidden_dim2': 583,
                  'gqmnn_hidden_dim3': 89, 'fpn_hidden_dim': 495, 'gat_ci_out': 154, 'gat_e_out_dim': 315,
                  'weight_decay': 0.005708, 'fpn_dropout': 0.09, 'fpn_out_dim': 190, 'gnn_dropout': 0.33, 'nheads': 3,
                  'ratio': 0.63, 'gat_layer1_out_dim': 69, 'h_delta': 0.52}
    elif args.task_name == 'bace':
        # hyperp = {'dropout': 0.640, 'lr': 0.001,
        #           'gqmnn_hidden_dim1': 665, 'gqmnn_hidden_dim2': 520,
        #           'gqmnn_hidden_dim3': 35, 'fpn_hidden_dim': 820, 'gat_ci_out': 355,
        #           'gat_e_out_dim': 310,
        #           'weight_decay': 0.00023}
        # args.fpn_dropout = 0.4
        # args.fpn_out_dim = 350
        # args.gnn_dropout = 0.5
        # args.nheads = 3
        # args.ratio = 0.7
        # args.gat_layer1_out_dim = 60
        # hyperp={'dropout': 0.4, 'lr': 0.0010740380218122185, 'gqmnn_hidden_dim1': 994, 'gqmnn_hidden_dim2': 726,
        #  'gqmnn_hidden_dim3': 235, 'fpn_hidden_dim': 735, 'gat_ci_out': 247, 'gat_e_out_dim': 215,
        #  'weight_decay': 0.0054443659020254175, 'fpn_dropout': 0.5, 'fpn_out_dim': 752, 'gnn_dropout': 0.1, 'nheads': 2,
        #  'ratio': 0.5, 'gat_layer1_out_dim': 182}
        # hyperp = {'dropout': 0.4, 'lr': 0.0011, 'gqmnn_hidden_dim1': 994, 'gqmnn_hidden_dim2': 726,
        #            'gqmnn_hidden_dim3': 235, 'fpn_hidden_dim': 735, 'gat_ci_out': 247, 'gat_e_out_dim': 215,
        #            'weight_decay': 0.0055, 'fpn_dropout': 0.5, 'fpn_out_dim': 752, 'gnn_dropout': 0.1, 'nheads': 2,
        #            'ratio': 0.5, 'gat_layer1_out_dim': 182}\
        if args.split_type == 'random':

            hyperp = {'dropout': 0.5, 'lr': 0.0009, 'gqmnn_hidden_dim1': 994, 'gqmnn_hidden_dim2': 726,
                      'gqmnn_hidden_dim3': 235, 'fpn_hidden_dim': 735, 'gat_ci_out': 247, 'gat_e_out_dim': 215,
                      'weight_decay': 0.006, 'fpn_dropout': 0.5, 'fpn_out_dim': 752, 'gnn_dropout': 0.1, 'nheads': 2,
                      'ratio': 0.5,
                      'gat_layer1_out_dim': 182}  # (0.8894392016594674, 0.025080829335318005, 0.884936703346596, 0.02315059450483383)
            # args.fpn_dropout = 0.4
            # args.fpn_out_dim = 350
            # args.gnn_dropout = 0.5
            # args.nheads = 3
            # args.ratio = 0.7
            # args.gat_layer1_out_dim = 60
            # hyperp={'dropout': 0.1, 'lr': 0.00296,'gqmnn_hidden_dim1': 582, 'gqmnn_hidden_dim2': 253,
            #                                        'gqmnn_hidden_dim3': 221, 'fpn_hidden_dim': 840, 'gat_ci_out': 101,
            #                                        'gat_e_out_dim': 64, 'weight_decay': 0.00885,
            #                                        'fpn_dropout': 0.60, 'fpn_out_dim': 544,
            #                                        'gnn_dropout': 0.5, 'nheads': 3, 'ratio': 0.8,
            #                                        'gat_layer1_out_dim': 104}
            #(0.8875608345257309, 0.02941527814862942, 0.8879589780853454, 0.02252404175063399)
            hyperp = {'dropout': 0.023, 'lr': 0.0008, 'gqmnn_hidden_dim1': 760,
                      'gqmnn_hidden_dim2': 950, 'gqmnn_hidden_dim3': 300, 'fpn_hidden_dim': 555, 'gat_ci_out': 23,
                      'gat_e_out_dim': 165, 'weight_decay': 0.003, 'fpn_dropout': 0.28,
                      'fpn_out_dim': 50, 'gnn_dropout': 0.034, 'nheads': 4, 'ratio': 0.24,
                      'gat_layer1_out_dim': 65}
            hyperp = {'dropout': 0.023, 'lr': 0.0008, 'gqmnn_hidden_dim1': 760,
                      'gqmnn_hidden_dim2': 950, 'gqmnn_hidden_dim3': 300, 'fpn_hidden_dim': 555, 'gat_ci_out': 23,
                      'gat_e_out_dim': 165, 'weight_decay': 0.00003, 'fpn_dropout': 0.28,
                      'fpn_out_dim': 50, 'gnn_dropout': 0.034, 'nheads': 4, 'ratio': 0.24,
                      'gat_layer1_out_dim': 65}
            # hyperp = {'dropout': 0.023, 'lr': 0.0008, 'gqmnn_hidden_dim1': 760,
            #           'gqmnn_hidden_dim2': 950, 'gqmnn_hidden_dim3': 300, 'fpn_hidden_dim': 555, 'gat_ci_out': 23,
            #           'gat_e_out_dim': 165, 'weight_decay': 0.003, 'fpn_dropout': 0.28,
            #           'fpn_out_dim': 50, 'gnn_dropout': 0.034, 'nheads': 8, 'ratio': 0.5,
            #           'gat_layer1_out_dim': 65}
            # (0.8921130854311317, 0.02476435105735386, 0.8987914616252383, 0.021007981491618484)
            hyperp = {'dropout': 0.05, 'lr': 0.0005, 'gqmnn_hidden_dim1': 760,
                      'gqmnn_hidden_dim2': 950, 'gqmnn_hidden_dim3': 300, 'fpn_hidden_dim': 555, 'gat_ci_out': 25,
                      'gat_e_out_dim': 165, 'weight_decay': 0.00003, 'fpn_dropout': 0.3,
                      'fpn_out_dim': 50, 'gnn_dropout': 0.035, 'nheads': 3, 'ratio': 0.25,
                      'gat_layer1_out_dim': 65}
            #(0.894364743418491, 0.024904510814954983, 0.8977947628695693, 0.021669489297191657)
            #(0.8875445179049224, 0.029680606976019253, 0.8974499646841888, 0.027263673292165065) NO GCI
            hyperp = {'dropout': 0.05, 'lr': 0.0005, 'gqmnn_hidden_dim1': 760,
                      'gqmnn_hidden_dim2': 950, 'gqmnn_hidden_dim3': 300, 'fpn_hidden_dim': 555, 'gat_ci_out': 25,
                      'gat_e_out_dim': 165, 'weight_decay': 0.00004, 'fpn_dropout': 0.3,
                      'fpn_out_dim': 50, 'gnn_dropout': 0.035, 'nheads': 3, 'ratio': 0.25,
                      'gat_layer1_out_dim': 65}

            args.labels = ['Class']
            hyperp['fp_dim']=700
            args.noise_rate = 0.0
        else:

            hyperp={'dropout': 0.5, 'lr': 0.00027, 'gqmnn_hidden_dim1': 192,
             'gqmnn_hidden_dim2': 947, 'gqmnn_hidden_dim3': 209, 'fpn_hidden_dim': 496, 'gat_ci_out': 129,
             'gat_e_out_dim': 83, 'weight_decay': 0.000378, 'fpn_dropout': 0.7,
             'fpn_out_dim': 367, 'gnn_dropout': 0.0, 'nheads': 4, 'ratio': 0.7, 'gat_layer1_out_dim': 52}
            # hyperp={'dropout': 0.5, 'lr': 0.0001, 'gqmnn_hidden_dim1': 192,
            #  'gqmnn_hidden_dim2': 947, 'gqmnn_hidden_dim3': 209, 'fpn_hidden_dim': 496, 'gat_ci_out': 129,
            #  'gat_e_out_dim': 83, 'weight_decay': 0.0004, 'fpn_dropout': 0.3,
            #  'fpn_out_dim': 367, 'gnn_dropout': 0.0, 'nheads': 4, 'ratio': 0.7, 'gat_layer1_out_dim': 52}
            args.labels = ['Class']

    elif args.task_name == 'esol':
        hyperp = {'dropout': 0.023, 'lr': 0.0008, 'gqmnn_hidden_dim1': 760,
                  'gqmnn_hidden_dim2': 950, 'gqmnn_hidden_dim3': 300, 'fpn_hidden_dim': 555, 'gat_ci_out': 23,
                  'gat_e_out_dim': 165, 'weight_decay': 0.003, 'fpn_dropout': 0.28,
                  'fpn_out_dim': 50, 'gnn_dropout': 0.034, 'nheads': 4, 'ratio': 0.24,
                  'gat_layer1_out_dim': 65}
        # (0.6469505902297394, 0.06454328293880346, 0.61790407, 0.057601415)
        # (0.6510888140741903, 0.0522685444855318, 0.6155637, 0.061640624) no_gci
        # (0.6593548594361626, 0.05517201910534334, 0.6365636, 0.08942519) no we
        #(0.6351350078410932, 0.05454292828184082, 0.62137663, 0.07380
        #
        hyperp = {'dropout': 0.023, 'lr': 0.0008, 'gqmnn_hidden_dim1': 760,
                  'gqmnn_hidden_dim2': 950, 'gqmnn_hidden_dim3': 300, 'fpn_hidden_dim': 555, 'gat_ci_out': 23,
                  'gat_e_out_dim': 165, 'weight_decay': 0.003, 'fpn_dropout': 0.28,
                  'fpn_out_dim': 50, 'gnn_dropout': 0.034, 'nheads': 5, 'ratio': 0.24,
                  'gat_layer1_out_dim': 65, 'h_delta': 1}
        args.labels = ['Class']
        args.dataset_type = 'regression'
    elif args.task_name=='pdbbind_c':
        hyperp = {'dropout': 0.6, 'lr': 0.002, 'gqmnn_hidden_dim1': 840, 'gqmnn_hidden_dim2': 510,
                  'gqmnn_hidden_dim3': 120, 'fpn_hidden_dim': 100, 'gat_ci_out': 150, 'gat_e_out_dim': 370,
                  'weight_decay': 0.1, 'fpn_dropout': 0.9, 'fpn_out_dim': 900, 'gnn_dropout': 0.1, 'nheads': 8,
                  'ratio': 0.6, 'gat_layer1_out_dim': 50, 'h_delta': 1}
        hyperp = {'dropout': 0.15, 'lr': 0.002899162604538241, 'gqmnn_hidden_dim1': 928, 'gqmnn_hidden_dim2': 760,
                  'gqmnn_hidden_dim3': 642, 'fpn_hidden_dim': 945, 'gat_ci_out': 499, 'gat_e_out_dim': 311,
                  'weight_decay': 4.971686748585462e-05, 'fpn_dropout': 0.9500000000000001, 'fpn_out_dim': 180,
                  'gnn_dropout': 0.47000000000000003, 'nheads': 4, 'ratio': 0.18000000000000002,
                  'gat_layer1_out_dim': 164, 'h_delta': 0.01}
        args.labels = ['Class']
        args.dataset_type = 'regression'
        args.noise_rate=0.0
    elif args.task_name=='lipo' :
       # (0.7283793828797307, 0.0441708787411006, 0.6746531, 0.022525856)
     hyperp = {'dropout': 0.3, 'lr': 0.001, 'gqmnn_hidden_dim1': 300,
               'gqmnn_hidden_dim2': 110, 'gqmnn_hidden_dim3': 32, 'dn_out_dim': 230, 'fpn_hidden_dim': 680,
               'gat_ci_out': 100,
               'gat_e_out_dim': 80,'h_delta':1}
     args.fpn_dropout = 0.6
     args.fpn_out_dim = 600
     args.gnn_dropout = 0.25
     args.nheads = 5
     args.ratio = 0.5
     args.gat_layer1_out_dim = 80
     args.labels = ['Class']
     args.dataset_type='regression'
     args.noise_rate = 0.0
     args.weight_decay =1e-6
     args.weight_decay = 1e-4
       #(0.6454674239428523, 0.02606563279661582, 0.61447775, 0.030288441)
     hyperp={'dropout': 0.0, 'lr': 3.729658666414416e-05, 'gqmnn_hidden_dim1': 1555, 'gqmnn_hidden_dim2': 740,
        'gqmnn_hidden_dim3': 282, 'fpn_hidden_dim': 1541, 'gat_ci_out': 421, 'gat_e_out_dim': 208,
        'weight_decay': 8.217707334010459e-05, 'fp_dim': 100, 'fpn_dropout': 0.0, 'fpn_out_dim': 1822,
        'gnn_dropout': 0.15, 'nheads': 5, 'ratio': 0.7, 'gat_layer1_out_dim': 23, 'h_delta': 0.78}
       #(0.6453871231255517, 0.031520305115519655, 0.6137955, 0.03456245)
     hyperp = {'dropout': 0.0, 'lr': 3.7e-05, 'gqmnn_hidden_dim1': 1555, 'gqmnn_hidden_dim2': 740,
                 'gqmnn_hidden_dim3': 282, 'fpn_hidden_dim': 1541, 'gat_ci_out': 421, 'gat_e_out_dim': 208,
                 'weight_decay': 8.2e-05, 'fp_dim': 2513, 'fpn_dropout': 0.0, 'fpn_out_dim': 1822,
                 'gnn_dropout': 0.15, 'nheads': 5, 'ratio': 0.7, 'gat_layer1_out_dim': 23, 'h_delta': 0.78}
       #(0.6431747010132997, 0.02798626991210092, 0.6096895, 0.028961712)  ratio 0.1 gci false
      # (0.6428162184631879, 0.031443262669387045, 0.61031264, 0.030978734) ratio 0.1 gci true
     hyperp = {'dropout': 0.0, 'lr': 3.7e-05, 'gqmnn_hidden_dim1': 1555, 'gqmnn_hidden_dim2': 740,
                 'gqmnn_hidden_dim3': 282, 'fpn_hidden_dim': 1541, 'gat_ci_out': 421, 'gat_e_out_dim': 208,
                 'weight_decay': 8.2e-05, 'fp_dim': 2513, 'fpn_dropout': 0.0, 'fpn_out_dim': 1822,
                 'gnn_dropout': 0.15, 'nheads': 4, 'ratio': 0.7, 'gat_layer1_out_dim': 50, 'h_delta': 0.78}
      # (0.6421129854163501, 0.033999272172019965, 0.61290395, 0.03235025)
     hyperp = {'dropout': 0.0, 'lr': 3.7e-05, 'gqmnn_hidden_dim1': 1555, 'gqmnn_hidden_dim2': 740,
                 'gqmnn_hidden_dim3': 282, 'fpn_hidden_dim': 1541, 'gat_ci_out': 421, 'gat_e_out_dim': 208,
                 'weight_decay': 8.2e-05, 'fp_dim': 2513, 'fpn_dropout': 0.0, 'fpn_out_dim': 1822,
                 'gnn_dropout': 0.15, 'nheads': 3, 'ratio': 0.1, 'gat_layer1_out_dim': 100, 'h_delta': 0.78}
       #(0.6443164028987975, 0.03218450069542516, 0.6083248, 0.02889966)
     args.with_gci = True
     hyperp = {'dropout': 0.0, 'lr': 3.7e-05, 'gqmnn_hidden_dim1': 1555, 'gqmnn_hidden_dim2': 740,
                 'gqmnn_hidden_dim3': 282, 'fpn_hidden_dim': 1541, 'gat_ci_out': 421, 'gat_e_out_dim': 208,
                 'weight_decay': 8.2e-05, 'fp_dim': 2513, 'fpn_dropout': 0.0, 'fpn_out_dim': 1822,
                 'gnn_dropout': 0.15, 'nheads': 2, 'ratio': 0.1, 'gat_layer1_out_dim': 50, 'h_delta': 0.78}
     args.with_gci = False
    #(0.6433650567901144, 0.03533613769674152, 0.6107906, 0.034893215)
     hyperp = {'dropout': 0.0, 'lr': 3.7e-05, 'gqmnn_hidden_dim1': 1555, 'gqmnn_hidden_dim2': 740,
                 'gqmnn_hidden_dim3': 282, 'fpn_hidden_dim': 1541, 'gat_ci_out': 421, 'gat_e_out_dim': 208,
                 'weight_decay': 8.2e-05, 'fp_dim': 2513, 'fpn_dropout': 0.0, 'fpn_out_dim': 1822,
                 'gnn_dropout': 0.15, 'nheads': 2, 'ratio': 0.1, 'gat_layer1_out_dim': 50, 'h_delta': 0.78}
    elif args.task_name=='clintox':
        if args.labels==['Class1']:
            #(0.8170572930489571, 0.07394946156700072, 0.8492800538398357, 0.08028564771095655)
            #(0.9134213452459772, 0.0386149562919899, 0.9249264928447861, 0.06027546539331027) 500
            # hyperp = {'dropout': 0.0, 'lr': 0.004088267716083919, 'gqmnn_hidden_dim1': 456, 'gqmnn_hidden_dim2': 937,
            #           'gqmnn_hidden_dim3': 10, 'fpn_hidden_dim': 725, 'gat_ci_out': 172, 'gat_e_out_dim': 301,
            #           'weight_decay': 0.005834729410336786, 'fpn_dropout': 0.5, 'fpn_out_dim': 839, 'gnn_dropout': 0.0,
            #           'nheads': 5, 'ratio': 0.2, 'gat_layer1_out_dim': 23}
            hyperp={'dropout': 0.30000000000000004, 'lr': 0.0037488362481704163, 'gqmnn_hidden_dim1': 630, 'gqmnn_hidden_dim2': 143, 'gqmnn_hidden_dim3': 297, 'fpn_hidden_dim': 406, 'gat_ci_out': 14, 'gat_e_out_dim': 491, 'weight_decay': 0.008357018458517738, 'fpn_dropout': 0.30000000000000004, 'fpn_out_dim': 318, 'gnn_dropout': 0.2, 'nheads': 4, 'ratio': 0.4, 'gat_layer1_out_dim': 189}
            hyperp = {'dropout': 0.5, 'lr': 0.00375, 'gqmnn_hidden_dim1': 630, 'gqmnn_hidden_dim2': 143,
                      'gqmnn_hidden_dim3': 297, 'fpn_hidden_dim': 406, 'gat_ci_out': 14, 'gat_e_out_dim': 491,
                      'weight_decay': 0.00836, 'fpn_dropout': 0.5, 'fpn_out_dim': 318, 'gnn_dropout': 0.4, 'nheads': 4,
                      'ratio': 0.4, 'gat_layer1_out_dim': 189}
        elif args.labels == ['Class2']:
            # (0.8544969497000728, 0.08080356701095107, 0.8597245243630418, 0.06820094260909583)
            # 500 (0.9372494600526002, 0.02894084056828119, 0.9431044368532266, 0.029001934800270798)
            args.weight_decay = 1e-6  # (0.8544969497000728, 0.08080356701095107, 0.8597245243630418, 0.06820094260909583)
            hyperp = {'dropout': 0.0, 'lr': 0.004088267716083919, 'gqmnn_hidden_dim1': 456, 'gqmnn_hidden_dim2': 937,
                      'gqmnn_hidden_dim3': 10, 'fpn_hidden_dim': 725, 'gat_ci_out': 172, 'gat_e_out_dim': 301,
                      'weight_decay': 0.005834729410336786, 'fpn_dropout': 0.5, 'fpn_out_dim': 839, 'gnn_dropout': 0.0,
                      'nheads': 5, 'ratio': 0.2, 'gat_layer1_out_dim': 23}
            hyperp = {'dropout': 0.0, 'lr': 0.0041, 'gqmnn_hidden_dim1': 456, 'gqmnn_hidden_dim2': 937,
                      'gqmnn_hidden_dim3': 10, 'fpn_hidden_dim': 725, 'gat_ci_out': 172, 'gat_e_out_dim': 301,
                      'weight_decay': 0.0058, 'fpn_dropout': 0.5, 'fpn_out_dim': 839, 'gnn_dropout': 0.0,
                      'nheads': 5, 'ratio': 0.2, 'gat_layer1_out_dim': 23}

    elif args.task_name =='hiv':
        if args.split_type == 'random':
            # hyperp = {'dropout': 0.3, 'lr': 0.001, 'gqmnn_hidden_dim1': 300,
            #           'gqmnn_hidden_dim2': 110, 'gqmnn_hidden_dim3': 32, 'dn_out_dim': 230, 'fpn_hidden_dim': 680,
            #           'gat_ci_out': 100,
            #           'gat_e_out_dim': 80}
            # args.fpn_dropout = 0.6
            # args.fpn_out_dim = 600
            # args.gnn_dropout = 0.25
            # args.nheads = 5
            # args.ratio = 0.5
            # args.gat_layer1_out_dim = 80
            # args.labels = ['Class']
            # args.noise_rate = 0.0
            # args.weight_decay = 1e-6

            hyperp = {'dropout': 0.4, 'lr': 0.0008, 'gqmnn_hidden_dim1': 190, 'gqmnn_hidden_dim2': 400,
                      'gqmnn_hidden_dim3': 280, 'fpn_hidden_dim': 170, 'gat_ci_out': 400, 'gat_e_out_dim': 290,
                      'weight_decay': 0.005, 'fpn_dropout': 0.5, 'fpn_out_dim': 921, 'gnn_dropout': 0.1,
                      'nheads': 2, 'ratio': 0.2, 'gat_layer1_out_dim': 153}
            hyperp = {'dropout': 0.6, 'lr': 0.0008, 'gqmnn_hidden_dim1': 190, 'gqmnn_hidden_dim2': 400,
                      'gqmnn_hidden_dim3': 280, 'fpn_hidden_dim': 170, 'gat_ci_out': 400, 'gat_e_out_dim': 290,
                      'weight_decay': 0.005, 'fpn_dropout': 0.6, 'fpn_out_dim': 920, 'gnn_dropout': 0.3,
                      'nheads': 2, 'ratio': 0.2, 'gat_layer1_out_dim': 150}
            # hyperp = {'dropout': 0.11, 'lr': 0.0009125463564569077, 'gqmnn_hidden_dim1': 841, 'gqmnn_hidden_dim2': 539,
            #           'gqmnn_hidden_dim3': 434, 'fpn_hidden_dim': 412, 'gat_ci_out': 144, 'gat_e_out_dim': 182,
            #           'weight_decay': 8.180296067429673e-05, 'fpn_dropout': 0.73, 'fpn_out_dim': 265,
            #           'gnn_dropout': 0.17, 'nheads': 6, 'ratio': 0.24000000000000002, 'gat_layer1_out_dim': 138}
            #(0.8276398026291742, 0.025272955477297892, 0.8383396592460443, 0.023225272310761838) 1800
            hyperp = {'dropout': 0.3, 'lr': 0.0009, 'gqmnn_hidden_dim1': 841, 'gqmnn_hidden_dim2': 539,
                      'gqmnn_hidden_dim3': 434, 'fpn_hidden_dim': 412, 'gat_ci_out': 144, 'gat_e_out_dim': 182,
                      'weight_decay': 8e-05, 'fpn_dropout': 0.73, 'fpn_out_dim': 265,
                      'gnn_dropout': 0.17, 'nheads': 6, 'ratio': 0.24, 'gat_layer1_out_dim': 138}
            #(0.8285314809847042, 0.02610877776015361, 0.835735763456384, 0.02138844816755521)
            hyperp = {'dropout': 0.3, 'lr': 0.0009, 'gqmnn_hidden_dim1': 841, 'gqmnn_hidden_dim2': 539,
                      'gqmnn_hidden_dim3': 434, 'fpn_hidden_dim': 412, 'gat_ci_out': 144, 'gat_e_out_dim': 182,
                      'weight_decay': 8e-05, 'fpn_dropout': 0.73, 'fpn_out_dim': 265,
                      'gnn_dropout': 0.4, 'nheads': 6, 'ratio': 0.24, 'gat_layer1_out_dim': 138}
            args.labels = ['Class']
        else:
            hyperp = {'dropout': 0.4, 'lr': 0.0008125, 'gqmnn_hidden_dim1': 189, 'gqmnn_hidden_dim2': 396,
                      'gqmnn_hidden_dim3': 283, 'fpn_hidden_dim': 174, 'gat_ci_out': 397, 'gat_e_out_dim': 288,
                      'weight_decay': 0.004426, 'fpn_dropout': 0.5, 'fpn_out_dim': 921, 'gnn_dropout': 0.1,
                      'nheads': 2, 'ratio': 0.2, 'gat_layer1_out_dim': 153}
            hyperp={'dropout': 0.0, 'lr': 0.004,
                                                    'gqmnn_hidden_dim1': 781, 'gqmnn_hidden_dim2': 436,
                                                    'gqmnn_hidden_dim3': 277, 'fpn_hidden_dim': 292, 'gat_ci_out': 476,
                                                    'gat_e_out_dim': 154, 'weight_decay': 0.000158,
                                                    'fpn_dropout': 0.3, 'fpn_out_dim': 508,
                                                    'gnn_dropout': 0.0, 'nheads': 2, 'ratio': 0.1,
                                                    'gat_layer1_out_dim': 172}
     #        hyperp= {'dropout': 0.4, 'lr': 0.00374, 'gqmnn_hidden_dim1': 762, 'gqmnn_hidden_dim2': 929,
     # 'gqmnn_hidden_dim3': 241, 'fpn_hidden_dim': 323, 'gat_ci_out': 184, 'gat_e_out_dim': 199,
     # 'weight_decay': 7.5e-05, 'fpn_dropout': 0.4, 'fpn_out_dim': 668, 'gnn_dropout': 0.2, 'nheads': 2,
     # 'ratio': 0.2, 'gat_layer1_out_dim': 171}
            #(0.7622750011617078, 0.027949939509374293, 0.7963159405886475, 0.047152741802909515)
            hyperp={'dropout': 0.4, 'lr': 0.003738, 'gqmnn_hidden_dim1': 762, 'gqmnn_hidden_dim2': 929,
             'gqmnn_hidden_dim3': 241, 'fpn_hidden_dim': 323, 'gat_ci_out': 184, 'gat_e_out_dim': 199,
             'weight_decay': 7.504e-05, 'fpn_dropout': 0.4, 'fpn_out_dim': 668, 'gnn_dropout': 0.2,
             'nheads': 2, 'ratio': 0.2, 'gat_layer1_out_dim': 171}
            hyperp = {'dropout': 0.4, 'lr': 0.0008, 'gqmnn_hidden_dim1': 190, 'gqmnn_hidden_dim2': 400,
                  'gqmnn_hidden_dim3': 280, 'fpn_hidden_dim': 170, 'gat_ci_out': 400, 'gat_e_out_dim': 290,
                  'weight_decay': 0.005, 'fpn_dropout': 0.5, 'fpn_out_dim': 921, 'gnn_dropout': 0.1,
                  'nheads': 2, 'ratio': 0.2, 'gat_layer1_out_dim': 153}
            hyperp={'dropout': 0.54, 'lr': 0.0014, 'gqmnn_hidden_dim1': 571, 'gqmnn_hidden_dim2': 199, 'gqmnn_hidden_dim3': 347, 'fpn_hidden_dim': 742, 'gat_ci_out': 220, 'gat_e_out_dim': 311, 'weight_decay': 7.8e-05, 'fpn_dropout': 0.31, 'fpn_out_dim': 758, 'gnn_dropout': 0.54, 'nheads': 3, 'ratio': 0.55, 'gat_layer1_out_dim': 121}
            hyperp={'dropout': 0.54, 'lr': 0.001455, 'gqmnn_hidden_dim1': 571, 'gqmnn_hidden_dim2': 199, 'gqmnn_hidden_dim3': 347, 'fpn_hidden_dim': 742, 'gat_ci_out': 220, 'gat_e_out_dim': 311, 'weight_decay': 1e-04, 'fpn_dropout': 0.31, 'fpn_out_dim': 758, 'gnn_dropout': 0.54, 'nheads': 3, 'ratio': 0.55, 'gat_layer1_out_dim': 121}
            hyperp = {'dropout': 0.54, 'lr': 0.0011, 'gqmnn_hidden_dim1': 182, 'gqmnn_hidden_dim2': 665,
              'gqmnn_hidden_dim3': 840, 'fpn_hidden_dim': 493, 'gat_ci_out': 415, 'gat_e_out_dim': 454,
              'weight_decay': 3e-05, 'fpn_dropout': 0.77, 'fpn_out_dim': 265, 'gnn_dropout': 0.89, 'nheads': 5,
              'ratio': 0.13, 'gat_layer1_out_dim': 198}
            hyperp={'dropout': 0.44, 'lr': 0.0011255814144526203, 'gqmnn_hidden_dim1': 986, 'gqmnn_hidden_dim2': 1043,
             'gqmnn_hidden_dim3': 599, 'fpn_hidden_dim': 1912, 'gat_ci_out': 302, 'gat_e_out_dim': 158,
             'weight_decay': 3.075115922927198e-05, 'fp_dim': 2200, 'fpn_dropout': 0.58, 'fpn_out_dim': 1598,
             'gnn_dropout': 0.46, 'nheads': 5, 'ratio': 0.6, 'gat_layer1_out_dim': 144}
            #(0.8268054545802475, 0.027877458610155833, 0.83626927503323, 0.03467300382137448)
            hyperp = {'dropout': 0.45, 'lr': 0.0011, 'gqmnn_hidden_dim1': 1000, 'gqmnn_hidden_dim2': 1050,
                      'gqmnn_hidden_dim3': 600, 'fpn_hidden_dim': 1900, 'gat_ci_out': 300, 'gat_e_out_dim': 160,
                      'weight_decay': 3e-05, 'fp_dim': 2200, 'fpn_dropout': 0.6, 'fpn_out_dim': 1600,
                      'gnn_dropout': 0.45, 'nheads': 5, 'ratio': 0.0, 'gat_layer1_out_dim': 150}
            args.ratio=0
    elif args.task_name=='pdbbind_f':

        #(1.2639996181052207, 0.025104512175069142, 1.2664349, 0.020176038)
        hyperp = {'dropout': 0.3, 'lr': 0.000129,
                  'gqmnn_hidden_dim1': 392, 'gqmnn_hidden_dim2': 838,
                  'gqmnn_hidden_dim3': 227, 'fpn_hidden_dim': 267, 'gat_ci_out': 71,
                  'gat_e_out_dim': 191, 'weight_decay': 0.00824,
                  'fpn_dropout': 0.5, 'fpn_out_dim': 996, 'gnn_dropout': 0.0, 'nheads': 4,
                  'ratio': 0.5, 'gat_layer1_out_dim': 166,'h_delta':1}

        args.labels = ['Class']
        args.dataset_type = 'regression'
    elif args.task_name=='sider':
        labels=[ 'Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders', 'Investigations', 'Musculoskeletal and connective tissue disorders',
                 'Gastrointestinal disorders', 'Social circumstances', 'Immune system disorders', 'Reproductive system and breast disorders',
                 'Neoplasms benign, malignant and unspecified (incl cysts and polyps)', 'General disorders and administration site conditions',
                 'Endocrine disorders', 'Surgical and medical procedures', 'Vascular disorders', 'Blood and lymphatic system disorders',
                 'Skin and subcutaneous tissue disorders', 'Congenital, familial and genetic disorders', 'Infections and infestations',
                 'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders', 'Renal and urinary disorders',
                 'Pregnancy, puerperium and perinatal conditions', 'Ear and labyrinth disorders', 'Cardiac disorders',
                 'Nervous system disorders', 'Injury, poisoning and procedural complications']

        if args.labels[0]==labels[0]:
           #Hepatobiliary disorders
           # (0.7350516301365937, 0.0364762785725114, 0.7238907353917275, 0.0213730700202813) 2513
           # (0.7843529903765553, 0.029285086528954243, 0.774225277168912, 0.018372294579891972) 400
            hyperp = {'dropout': 0.7, 'lr': 0.0014, 'gqmnn_hidden_dim1': 113, 'gqmnn_hidden_dim2': 842,
                      'gqmnn_hidden_dim3': 292, 'fpn_hidden_dim': 612, 'gat_ci_out': 429, 'gat_e_out_dim': 466,
                      'weight_decay': 0.0017, 'fpn_dropout': 0.6, 'fpn_out_dim': 232, 'gnn_dropout': 0.9,
                      'nheads': 3, 'ratio': 0.7, 'gat_layer1_out_dim': 191}
        elif args.labels[0]==labels[1]:#Metabolism and nutrition disorders

            #(0.7379165486707604, 0.05208859676446278, 0.7107225939361944, 0.058624813094040856) 600

            #(0.6542556867266385, 0.06687783765128137, 0.6439238132334769, 0.040194980192620196)2513

            hyperp = {'dropout': 0.7, 'lr': 0.0014, 'gqmnn_hidden_dim1': 113, 'gqmnn_hidden_dim2': 842,
                      'gqmnn_hidden_dim3': 292, 'fpn_hidden_dim': 612, 'gat_ci_out': 429, 'gat_e_out_dim': 466,
                      'weight_decay': 0.0017, 'fpn_dropout': 0.6, 'fpn_out_dim': 232, 'gnn_dropout': 0.9,
                      'nheads': 3, 'ratio': 0.7, 'gat_layer1_out_dim': 191}
        elif args.labels[0] == labels[2]:#Product issues

            #(0.3598339536923241, 0.17211339294623454, 0.6520759416148587, 0.22743020566155125) 2513
            #(0.8550609338193409, 0.09778207600675241, 0.8231291116693618, 0.10631688072718444) 500
            hyperp = {'dropout': 0.8, 'lr': 0.008, 'gqmnn_hidden_dim1': 205, 'gqmnn_hidden_dim2': 198, 'gqmnn_hidden_dim3': 106, 'fpn_hidden_dim': 203, 'gat_ci_out': 33, 'gat_e_out_dim': 27, 'weight_decay': 0.003, 'fpn_dropout': 0.1, 'fpn_out_dim': 668, 'gnn_dropout': 1.0, 'nheads': 1, 'ratio': 0.5, 'gat_layer1_out_dim': 125}
        elif args.labels[0] == labels[3]:#Eye disorders
            #(0.6799746314494107, 0.04048748311674906, 0.6601242198017641, 0.05129656890529838) 2513
            #(0.7851010064165097, 0.024526862307439317, 0.756319671505017, 0.039183075626557234) 600
            hyperp={'dropout': 0.3, 'lr': 0.001,
                                                    'gqmnn_hidden_dim1': 194, 'gqmnn_hidden_dim2': 147,
                                                    'gqmnn_hidden_dim3': 123, 'fpn_hidden_dim': 421, 'gat_ci_out': 426,
                                                    'gat_e_out_dim': 425, 'weight_decay': 0.000122,
                                                    'fpn_dropout': 0.6, 'fpn_out_dim': 782,
                                                    'gnn_dropout': 0.2, 'nheads': 1, 'ratio': 0.8,
                                                    'gat_layer1_out_dim': 69}
        elif args.labels[0] == labels[4]:#Investigations

            #(0.7537448487139127, 0.05700535071065069, 0.7702559833693992, 0.05740612971747046) 500
            #(0.6807368756227504, 0.05220190133002632, 0.6744364365194071, 0.039599551292703164) 2513
            hyperp = {'dropout': 0.7, 'lr': 0.0014, 'gqmnn_hidden_dim1': 113, 'gqmnn_hidden_dim2': 842,
                      'gqmnn_hidden_dim3': 292, 'fpn_hidden_dim': 612, 'gat_ci_out': 429, 'gat_e_out_dim': 466,
                      'weight_decay': 0.0017, 'fpn_dropout': 0.6, 'fpn_out_dim': 232, 'gnn_dropout': 0.9,
                      'nheads': 3, 'ratio': 0.7, 'gat_layer1_out_dim': 191}
        elif args.labels[0] == labels[5]:#Musculoskeletal

            #(0.6354312502111512, 0.046811821351125146, 0.6655128244970449, 0.0362531197149215) 2513
            #(0.6776703576293516, 0.05285536060989163, 0.7143100092275696, 0.03296060157044929) 600
            hyperp={'dropout': 0.3, 'lr': 0.00025,
                                                    'gqmnn_hidden_dim1': 763, 'gqmnn_hidden_dim2': 61,
                                                    'gqmnn_hidden_dim3': 91, 'fpn_hidden_dim': 372, 'gat_ci_out': 481,
                                                    'gat_e_out_dim': 500, 'weight_decay': 0.00241,
                                                    'fpn_dropout': 0.0, 'fpn_out_dim': 66, 'gnn_dropout': 0.0,
                                                    'nheads': 1, 'ratio': 0.9, 'gat_layer1_out_dim': 79}



        elif args.labels[0] == labels[6]:#Gastrointestinal
            #(0.7499411163239278, 0.0468268781965862, 0.7868760542243984, 0.05779609443543357) 2513
            #(0.8836762963398602, 0.037668101330036806, 0.9048277755561186, 0.03138404341549287) 500
            hyperp = {'dropout': 0.6, 'lr': 0.001,
                      'gqmnn_hidden_dim1': 194, 'gqmnn_hidden_dim2': 147,
                      'gqmnn_hidden_dim3': 123, 'fpn_hidden_dim': 421, 'gat_ci_out': 426,
                      'gat_e_out_dim': 425, 'weight_decay': 0.00013,
                      'fpn_dropout': 0.6, 'fpn_out_dim': 782,
                      'gnn_dropout': 0.4, 'nheads': 1, 'ratio': 0.8,
                      'gat_layer1_out_dim': 69}
        elif args.labels[0] == labels[7]:#social
            #(0.624291064514612, 0.06289897280970044, 0.6153029034292372, 0.07410148384218183) 2513
            #(0.7633079953365779, 0.05181989899457029, 0.786642246106027, 0.05993610222142707) 200
            hyperp = {'dropout': 0.6, 'lr': 0.0039, 'gqmnn_hidden_dim1': 753, 'gqmnn_hidden_dim2': 773,
                      'gqmnn_hidden_dim3': 196, 'fpn_hidden_dim': 404, 'gat_ci_out': 78, 'gat_e_out_dim': 281,
                      'weight_decay': 0.00076, 'fpn_dropout': 0.2, 'fpn_out_dim': 468, 'gnn_dropout': 0.0, 'nheads': 2,
                      'ratio': 0.2, 'gat_layer1_out_dim': 133}

        elif args.labels[0] == labels[8]:#Immune system disorders
            #(0.6115061414837764, 0.052575026647302445, 0.6148458433839618, 0.05486730526624417)2513
            #(0.7306252647976645, 0.032536858544533635, 0.7135144922156328, 0.06386557549399068) 500
            hyperp={'dropout': 0.5, 'lr': 0.002, 'gqmnn_hidden_dim1': 406, 'gqmnn_hidden_dim2': 975,
             'gqmnn_hidden_dim3': 154, 'fpn_hidden_dim': 468, 'gat_ci_out': 398, 'gat_e_out_dim': 224,
             'weight_decay': 0.0051, 'fpn_dropout': 0.4, 'fpn_out_dim': 607,
             'gnn_dropout': 0.7, 'nheads': 1, 'ratio': 0.7,
             'gat_layer1_out_dim': 112}


        elif args.labels[0] == labels[9]:#Reproductive system and breast disorders

            #(0.7184004300393622, 0.04141101324899422, 0.7442275912106618, 0.04117636095635182)
            hyperp = {'dropout': 0.6, 'lr': 0.001,
                      'gqmnn_hidden_dim1': 194, 'gqmnn_hidden_dim2': 147,
                      'gqmnn_hidden_dim3': 123, 'fpn_hidden_dim': 421, 'gat_ci_out': 426,
                      'gat_e_out_dim': 425, 'weight_decay': 0.00013,
                      'fpn_dropout': 0.6, 'fpn_out_dim': 782,
                      'gnn_dropout': 0.4, 'nheads': 1, 'ratio': 0.8,
                      'gat_layer1_out_dim': 69}


        elif args.labels[0] == labels[10]:  #  Neoplasms benign, malignant and unspecified (incl cysts and polyps)
            # (0.6991596673761321, 0.05926625073042977, 0.7246092228621619, 0.0490528205111666) 2513
            #(0.7687108391880381, 0.044993350559076266, 0.7784168740991151, 0.057995046309322666) 400
            hyperp = {'dropout': 0.8, 'lr': 0.0003, 'gqmnn_hidden_dim1': 532, 'gqmnn_hidden_dim2': 831,
                      'gqmnn_hidden_dim3': 192, 'fpn_hidden_dim': 431, 'gat_ci_out': 24, 'gat_e_out_dim': 32,
                      'weight_decay': 1.1e-05, 'fpn_dropout': 0.3, 'fpn_out_dim': 827, 'gnn_dropout': 0.6, 'nheads': 2,
                      'ratio': 0.3, 'gat_layer1_out_dim': 252}
        elif args.labels[0] == labels[11]:  # General disorders and administration site conditions
            #(0.5663702241753767, 0.07083850871500295, 0.6930801474448132, 0.09160903828510374) 2513
            #(0.8053272075446276, 0.05078144051849975, 0.856600031571579, 0.044519755571803966) 500
            hyperp = {'dropout': 0.1, 'lr': 0.0012, 'gqmnn_hidden_dim1': 101, 'gqmnn_hidden_dim2': 972,
                      'gqmnn_hidden_dim3': 249, 'fpn_hidden_dim': 669, 'gat_ci_out': 277, 'gat_e_out_dim': 271,
                      'weight_decay': 0.000552, 'fpn_dropout': 0.7, 'fpn_out_dim': 440, 'gnn_dropout': 0.1, 'nheads': 2,
                      'ratio': 0.6, 'gat_layer1_out_dim': 40}
        elif args.labels[0] == labels[12]:  # Endocrine disorders

            # (0.6893332481282048, 0.030528938214538014, 0.7260638275703645, 0.03348554704139729)
            hyperp = {'dropout': 0.5, 'lr': 0.0022, 'gqmnn_hidden_dim1': 800, 'gqmnn_hidden_dim2': 636,
                      'gqmnn_hidden_dim3': 224, 'fpn_hidden_dim': 692, 'gat_ci_out': 13, 'gat_e_out_dim': 129,
                      'weight_decay': 0.00027, 'fpn_dropout': 0.7, 'fpn_out_dim': 731, 'gnn_dropout': 0.9, 'nheads': 2,
                      'ratio': 0.9, 'gat_layer1_out_dim': 255}
        elif args.labels[0] == labels[13]:  # Surgical and medical procedures
            #(0.5673264778492698, 0.048411682001051705, 0.5836596497759332, 0.11436534124447593) 2513
            #(0.6686294029248606, 0.11849001251570801, 0.7173222957192769, 0.10657976886222031) 700
            hyperp = {'dropout': 0.7, 'lr': 0.00237, 'gqmnn_hidden_dim1': 82, 'gqmnn_hidden_dim2': 226,
                      'gqmnn_hidden_dim3': 141, 'fpn_hidden_dim': 279, 'gat_ci_out': 163, 'gat_e_out_dim': 61,
                      'weight_decay': 0.00456, 'fpn_dropout': 0.6, 'fpn_out_dim': 183, 'gnn_dropout': 1.0, 'nheads': 2,
                      'ratio': 0.8, 'gat_layer1_out_dim': 87}
        elif args.labels[0] == labels[14]:  # Vascular disorders

            # (0.6518685553421651, 0.03309972392755309, 0.6723320053787595, 0.05342630053033473)
            hyperp = {'dropout': 0.5, 'lr': 0.0022, 'gqmnn_hidden_dim1': 800, 'gqmnn_hidden_dim2': 636,
                      'gqmnn_hidden_dim3': 224, 'fpn_hidden_dim': 692, 'gat_ci_out': 13, 'gat_e_out_dim': 129,
                      'weight_decay': 0.00027, 'fpn_dropout': 0.7, 'fpn_out_dim': 731, 'gnn_dropout': 0.9, 'nheads': 2,
                      'ratio': 0.9, 'gat_layer1_out_dim': 255}
        elif args.labels[0] == labels[15]:  # Blood and lymphatic system disorders

            # (0.7477977422324794, 0.028531192191607765, 0.7562311751695461, 0.04497108924920152)
            hyperp = {'dropout': 0.5, 'lr': 0.0022, 'gqmnn_hidden_dim1': 800, 'gqmnn_hidden_dim2': 636,
                      'gqmnn_hidden_dim3': 224, 'fpn_hidden_dim': 692, 'gat_ci_out': 13, 'gat_e_out_dim': 129,
                      'weight_decay': 0.00027, 'fpn_dropout': 0.7, 'fpn_out_dim': 731, 'gnn_dropout': 0.9, 'nheads': 2,
                      'ratio': 0.9, 'gat_layer1_out_dim': 255}
        elif args.labels[0] == labels[16]:  # 'Skin and subcutaneous tissue disorders'

            # (0.6886137842821161, 0.08296981782711277, 0.7253976310239421, 0.05872052539457172)
            hyperp = {'dropout': 0.6, 'lr': 0.0039, 'gqmnn_hidden_dim1': 753, 'gqmnn_hidden_dim2': 773,
                      'gqmnn_hidden_dim3': 196, 'fpn_hidden_dim': 404, 'gat_ci_out': 78, 'gat_e_out_dim': 281,
                      'weight_decay': 0.00076, 'fpn_dropout': 0.2, 'fpn_out_dim': 468, 'gnn_dropout': 0.0, 'nheads': 2,
                      'ratio': 0.2, 'gat_layer1_out_dim': 133}
        elif args.labels[0] == labels[17]:  # Congenital, familial and genetic disorders

            # (0.5815109304026066, 0.062139184081756116, 0.5747263663792535, 0.053391035475139806)
            hyperp = {'dropout': 0.5, 'lr': 0.0022, 'gqmnn_hidden_dim1': 800, 'gqmnn_hidden_dim2': 636,
                      'gqmnn_hidden_dim3': 224, 'fpn_hidden_dim': 692, 'gat_ci_out': 13, 'gat_e_out_dim': 129,
                      'weight_decay': 0.00027, 'fpn_dropout': 0.7, 'fpn_out_dim': 731, 'gnn_dropout': 0.9, 'nheads': 2,
                      'ratio': 0.9, 'gat_layer1_out_dim': 255}
        elif args.labels[0] == labels[18]:  # Infections and infestations

            hyperp={'dropout': 0.1, 'lr': 0.0003, 'gqmnn_hidden_dim1': 971, 'gqmnn_hidden_dim2': 568, 'gqmnn_hidden_dim3': 238, 'fpn_hidden_dim': 1000, 'gat_ci_out': 58, 'gat_e_out_dim': 240, 'weight_decay': 0.004, 'fpn_dropout': 0.8, 'fpn_out_dim': 935, 'gnn_dropout': 0.3, 'nheads': 4, 'ratio': 0.5, 'gat_layer1_out_dim': 19}

        elif args.labels[0] == labels[19]:  #Respiratory, thoracic and mediastinal disorders


            hyperp = {'dropout': 0.3, 'lr': 0.001,
                      'gqmnn_hidden_dim1': 194, 'gqmnn_hidden_dim2': 147,
                      'gqmnn_hidden_dim3': 123, 'fpn_hidden_dim': 421, 'gat_ci_out': 426,
                      'gat_e_out_dim': 425, 'weight_decay': 0.000122,
                      'fpn_dropout': 0.6, 'fpn_out_dim': 782,
                      'gnn_dropout': 0.2, 'nheads': 1, 'ratio': 0.8,
                      'gat_layer1_out_dim': 69}

        elif args.labels[0] == labels[20]:  # Psychiatric disorders
            # (0.710526767071096, 0.03311711505654963, 0.6934436535936531, 0.03392550845792972)
            hyperp = {'dropout': 0.1, 'lr': 0.0012, 'gqmnn_hidden_dim1': 101, 'gqmnn_hidden_dim2': 972,
                      'gqmnn_hidden_dim3': 249, 'fpn_hidden_dim': 669, 'gat_ci_out': 277, 'gat_e_out_dim': 271,
                      'weight_decay': 0.000552, 'fpn_dropout': 0.7, 'fpn_out_dim': 440, 'gnn_dropout': 0.1, 'nheads': 2,
                      'ratio': 0.6, 'gat_layer1_out_dim': 40}
        elif args.labels[0] == labels[21]:  # Renal and urinary disorders
            # (0.6712111079536841, 0.03734757523893864, 0.7062896629937611, 0.027560278735368614)
            hyperp = {'dropout': 0.1, 'lr': 0.0012, 'gqmnn_hidden_dim1': 101, 'gqmnn_hidden_dim2': 972,
                      'gqmnn_hidden_dim3': 249, 'fpn_hidden_dim': 669, 'gat_ci_out': 277, 'gat_e_out_dim': 271,
                      'weight_decay': 0.000552, 'fpn_dropout': 0.7, 'fpn_out_dim': 440, 'gnn_dropout': 0.1, 'nheads': 2,
                      'ratio': 0.6, 'gat_layer1_out_dim': 40}
        elif args.labels[0] == labels[22]:  # Pregnancy, puerperium and perinatal conditions

            hyperp = {'dropout': 0.7, 'lr': 0.00169, 'gqmnn_hidden_dim1': 324,
                      'gqmnn_hidden_dim2': 119, 'gqmnn_hidden_dim3': 13, 'fpn_hidden_dim': 430, 'gat_ci_out': 499,
                      'gat_e_out_dim': 325, 'weight_decay': 0.00176, 'fpn_dropout': 0.8, 'fpn_out_dim': 54,
                      'gnn_dropout': 0.8, 'nheads': 2, 'ratio': 0.8, 'gat_layer1_out_dim': 52}
        elif args.labels[0] == labels[23]:  # Ear and labyrinth disorders


            hyperp = {'dropout': 0.6, 'lr': 0.0004, 'gqmnn_hidden_dim1': 568,
                      'gqmnn_hidden_dim2': 802, 'gqmnn_hidden_dim3': 194, 'fpn_hidden_dim': 555, 'gat_ci_out': 85,
                      'gat_e_out_dim': 266, 'weight_decay': 0.002, 'fpn_dropout': 0.6,
                      'fpn_out_dim': 246, 'gnn_dropout': 1.0, 'nheads': 2, 'ratio': 0.1, 'gat_layer1_out_dim': 150}


        elif args.labels[0] == labels[24]:  # Cardiac disorders
           #(0.7677497603589203, 0.04378772414189478, 0.7809736078417453, 0.02407794603532327) 500
            hyperp = {'dropout': 0.1, 'lr': 0.0012, 'gqmnn_hidden_dim1': 101, 'gqmnn_hidden_dim2': 972,
                      'gqmnn_hidden_dim3': 249, 'fpn_hidden_dim': 669, 'gat_ci_out': 277, 'gat_e_out_dim': 271,
                      'weight_decay': 0.000552, 'fpn_dropout': 0.7, 'fpn_out_dim': 440, 'gnn_dropout': 0.1, 'nheads': 2,
                      'ratio': 0.6, 'gat_layer1_out_dim': 40}
        elif args.labels[0] == labels[25]:  # Nervous system disorders

            hyperp = {'dropout': 0.3, 'lr': 0.00025,
                      'gqmnn_hidden_dim1': 763, 'gqmnn_hidden_dim2': 61,
                      'gqmnn_hidden_dim3': 91, 'fpn_hidden_dim': 372, 'gat_ci_out': 481,
                      'gat_e_out_dim': 500, 'weight_decay': 0.00241,
                      'fpn_dropout': 0.0, 'fpn_out_dim': 66, 'gnn_dropout': 0.0,
                      'nheads': 1, 'ratio': 0.9, 'gat_layer1_out_dim': 79}
        elif args.labels[0] == labels[26]:  # Injury, poisoning and procedural complications

            hyperp = {'dropout': 0.6, 'lr': 0.0039, 'gqmnn_hidden_dim1': 753, 'gqmnn_hidden_dim2': 773,
                      'gqmnn_hidden_dim3': 196, 'fpn_hidden_dim': 404, 'gat_ci_out': 78, 'gat_e_out_dim': 281,
                      'weight_decay': 0.00076, 'fpn_dropout': 0.2, 'fpn_out_dim': 468, 'gnn_dropout': 0.0, 'nheads': 2,
                      'ratio': 0.2, 'gat_layer1_out_dim': 133}


    elif args.task_name=='muv':
        args.dataset_type = 'classification'
        hyperp = {'dropout': 0.3, 'lr': 0.001, 'gqmnn_hidden_dim1': 300,
                  'gqmnn_hidden_dim2': 110, 'gqmnn_hidden_dim3': 32, 'dn_out_dim': 230, 'fpn_hidden_dim': 680,
                  'gat_ci_out': 100,
                  'gat_e_out_dim': 80}
        args.fpn_dropout = 0.6
        args.fpn_out_dim = 600
        args.gnn_dropout = 0.25
        args.nheads = 5
        args.ratio = 0.5
        args.gat_layer1_out_dim = 80
        args.noise_rate = 0.0
        args.weight_decay = 1e-6
        args.classification_type='PRC'
        args.labels=['MUV-548']
    elif args.task_name == 'mda_mb_453':
        hyperp = {'dropout': 0.3, 'lr': 0.00025,
                  'gqmnn_hidden_dim1': 763, 'gqmnn_hidden_dim2': 61,
                  'gqmnn_hidden_dim3': 91, 'fpn_hidden_dim': 372, 'gat_ci_out': 481,
                  'gat_e_out_dim': 500, 'weight_decay': 0.00241,
                  'fpn_dropout': 0.0, 'fpn_out_dim': 66, 'gnn_dropout': 0.0,
                  'nheads': 1, 'ratio': 0.9, 'gat_layer1_out_dim': 79}
        hyperp={'dropout': 0.4207, 'lr': 0.00272, 'gqmnn_hidden_dim1': 926, 'gqmnn_hidden_dim2': 941,
         'gqmnn_hidden_dim3': 700, 'fpn_hidden_dim': 432, 'gat_ci_out': 96, 'gat_e_out_dim': 200,
         'weight_decay': 0.000200, 'fpn_dropout': 0.4281, 'fpn_out_dim': 987,
         'gnn_dropout': 0.5440, 'nheads': 3, 'ratio': 0.2, 'gat_layer1_out_dim': 87}
        hyperp={'dropout': 0.420, 'lr': 0.002718, 'gqmnn_hidden_dim1': 926, 'gqmnn_hidden_dim2': 941, 'gqmnn_hidden_dim3': 699, 'fpn_hidden_dim': 432, 'gat_ci_out': 96, 'gat_e_out_dim': 200, 'weight_decay': 0.0002, 'fpn_dropout': 0.428, 'fpn_out_dim': 987, 'gnn_dropout': 0.544, 'nheads': 3, 'ratio': 0.19, 'gat_layer1_out_dim': 87}
       #(0.852402169597112, 0.03869152883102528, 0.8833382660013095, 0.07574268781329217)
        hyperp={'dropout': 0.5, 'lr': 0.002718, 'gqmnn_hidden_dim1': 926, 'gqmnn_hidden_dim2': 941, 'gqmnn_hidden_dim3': 699, 'fpn_hidden_dim': 432, 'gat_ci_out': 96, 'gat_e_out_dim': 200, 'weight_decay': 0.0002, 'fpn_dropout': 0.428, 'fpn_out_dim': 987, 'gnn_dropout': 0.544, 'nheads': 3, 'ratio': 0.19, 'gat_layer1_out_dim': 87}
        #(0.8489076757448906, 0.04320882035404348, 0.8944977383372524, 0.053867343351032874)
        hyperp={'dropout': 0.7, 'lr': 0.002, 'gqmnn_hidden_dim1': 926, 'gqmnn_hidden_dim2': 941, 'gqmnn_hidden_dim3': 699, 'fpn_hidden_dim': 432, 'gat_ci_out': 96, 'gat_e_out_dim': 200, 'weight_decay': 0.0002, 'fpn_dropout': 0.428, 'fpn_out_dim': 987, 'gnn_dropout': 0.544, 'nheads': 3, 'ratio': 0.19, 'gat_layer1_out_dim': 87}
        #(0.8452733910572464, 0.04562017341667066, 0.8846010400629967, 0.054364991355393216)
        hyperp={'dropout': 0.7, 'lr': 0.002, 'gqmnn_hidden_dim1': 926, 'gqmnn_hidden_dim2': 941, 'gqmnn_hidden_dim3': 699, 'fpn_hidden_dim': 432, 'gat_ci_out': 96, 'gat_e_out_dim': 200, 'weight_decay': 0.0003, 'fpn_dropout': 0.428, 'fpn_out_dim': 987, 'gnn_dropout': 0.544, 'nheads': 3, 'ratio': 0.19, 'gat_layer1_out_dim': 87}
        hyperp={'dropout': 0.55, 'lr': 0.002718, 'gqmnn_hidden_dim1': 926, 'gqmnn_hidden_dim2': 941, 'gqmnn_hidden_dim3': 699, 'fpn_hidden_dim': 432, 'gat_ci_out': 96, 'gat_e_out_dim': 200, 'weight_decay': 0.0002, 'fpn_dropout': 0.428, 'fpn_out_dim': 987, 'gnn_dropout': 0.544, 'nheads': 3, 'ratio': 0.19, 'gat_layer1_out_dim': 87}
        hyperp={'dropout': 0.5, 'lr': 0.002718, 'gqmnn_hidden_dim1': 926, 'gqmnn_hidden_dim2': 941, 'gqmnn_hidden_dim3': 699, 'fpn_hidden_dim': 432, 'gat_ci_out': 96, 'gat_e_out_dim': 200, 'weight_decay': 0.0002, 'fpn_dropout': 0.428, 'fpn_out_dim': 987, 'gnn_dropout': 0.544, 'nheads': 3, 'ratio': 0.19, 'gat_layer1_out_dim': 87}
    elif args.task_name == 'mda_mb_361':
        hyperp = {'dropout': 0.5, 'lr': 0.005, 'gqmnn_hidden_dim1': 700, 'gqmnn_hidden_dim2': 415,
                  'gqmnn_hidden_dim3': 180, 'fpn_hidden_dim': 335, 'gat_ci_out': 441, 'gat_e_out_dim': 214,
                  'weight_decay': 0.0033, 'fpn_dropout': 0.62, 'fpn_out_dim': 755, 'gnn_dropout': 0.11, 'nheads': 2,
                  'ratio': 0.51, 'gat_layer1_out_dim': 200}
    elif args.task_name == 'hbl_100':
        hyperp = {'dropout': 0.5, 'lr': 0.004, 'gqmnn_hidden_dim1': 700, 'gqmnn_hidden_dim2': 415,
                  'gqmnn_hidden_dim3': 180, 'fpn_hidden_dim': 335, 'gat_ci_out': 440, 'gat_e_out_dim': 200,
                  'weight_decay': 0.003, 'fpn_dropout': 0.6, 'fpn_out_dim': 755, 'gnn_dropout': 0.3, 'nheads': 2,
                  'ratio': 0.5, 'gat_layer1_out_dim': 200}
    elif args.task_name == 'bt_474':

        # (0.8800169642420375, 0.039710727209641396, 0.887910130168583, 0.038033814767733445)
        hyperp = {
            'dropout': 0.6, 'lr': 0.003, 'gqmnn_hidden_dim1': 861, 'gqmnn_hidden_dim2': 407,
            'gqmnn_hidden_dim3': 250,
            'fpn_hidden_dim': 711, 'gat_ci_out': 330, 'gat_e_out_dim': 331, 'weight_decay': 0.0025,
            'fpn_dropout': 0.8, 'fpn_out_dim': 439, 'gnn_dropout': 0.8, 'nheads': 2,
            'ratio': 0.5, 'gat_layer1_out_dim': 76}
    elif args.task_name == 'hs_578t':
        hyperp = {'dropout': 0.39, 'lr': 0.0023, 'gqmnn_hidden_dim1': 304, 'gqmnn_hidden_dim2': 572,
                  'gqmnn_hidden_dim3': 255, 'fpn_hidden_dim': 541, 'gat_ci_out': 139, 'gat_e_out_dim': 232,
                  'weight_decay': 0.00056, 'fpn_dropout': 0.79, 'fpn_out_dim': 615, 'gnn_dropout': 0.42, 'nheads': 1,
                  'ratio': 0.71, 'gat_layer1_out_dim': 139}
    elif args.task_name == 'bcap37':

        hyperp = {'dropout': 0.35, 'lr': 0.005, 'gqmnn_hidden_dim1': 384, 'gqmnn_hidden_dim2': 953,
                  'gqmnn_hidden_dim3': 697, 'fpn_hidden_dim': 432, 'gat_ci_out': 244, 'gat_e_out_dim': 387,
                  'weight_decay': 0.0023, 'fpn_dropout': 0.79, 'fpn_out_dim': 277, 'gnn_dropout': 0.37, 'nheads': 4,
                  'ratio': 0.43, 'gat_layer1_out_dim': 94}
    elif args.task_name == 'mda_mb_435':

        hyperp = {'dropout': 0.61, 'lr': 0.00032, 'gqmnn_hidden_dim1': 964, 'gqmnn_hidden_dim2': 122,
                 'gqmnn_hidden_dim3': 76, 'fpn_hidden_dim': 486, 'gat_ci_out': 293, 'gat_e_out_dim': 96,
                 'weight_decay': 0.0002, 'fpn_dropout': 0.45, 'fpn_out_dim': 140, 'gnn_dropout': 0.5, 'nheads': 1,
                 'ratio': 0.85, 'gat_layer1_out_dim': 88}

    elif args.task_name == 'sk_br_3':
        #(0.8602992721071887, 0.027749726175144224, 0.8453976492871345, 0.03189311774515939)
        hyperp = {'dropout': 0.5, 'lr': 0.0027, 'gqmnn_hidden_dim1': 926, 'gqmnn_hidden_dim2': 941,
                  'gqmnn_hidden_dim3': 699, 'fpn_hidden_dim': 432, 'gat_ci_out': 96, 'gat_e_out_dim': 200,
                  'weight_decay': 0.0002, 'fpn_dropout': 0.428, 'fpn_out_dim': 987, 'gnn_dropout': 0.544, 'nheads': 3,
                  'ratio': 0.19, 'gat_layer1_out_dim': 87}
    elif args.task_name == 'bt_20':
        # (0.8658913229872983, 0.07354493607294435, 0.8449840682193624, 0.06853109565419413)
        hyperp = {'dropout': 0.4, 'lr': 0.0008, 'gqmnn_hidden_dim1': 484, 'gqmnn_hidden_dim2': 821,
                  'gqmnn_hidden_dim3': 342, 'fpn_hidden_dim': 91, 'gat_ci_out': 410, 'gat_e_out_dim': 174,
                  'weight_decay': 0.014, 'fpn_dropout': 0.77, 'fpn_out_dim': 463, 'gnn_dropout': 0.45, 'nheads': 5,
                  'ratio': 0.52, 'gat_layer1_out_dim': 103}
    elif args.task_name == 't_47d':
        # (0.847415180149629, 0.019622738119805288, 0.8436780087902876, 0.03717008266789315)
        hyperp = {'dropout': 0.5, 'lr': 0.002, 'gqmnn_hidden_dim1': 900, 'gqmnn_hidden_dim2': 950,
                  'gqmnn_hidden_dim3': 700, 'fpn_hidden_dim': 430, 'gat_ci_out': 100, 'gat_e_out_dim': 200,
                  'weight_decay': 0.0002, 'fpn_dropout': 0.400, 'fpn_out_dim': 1000, 'gnn_dropout': 0.55, 'nheads': 3,
                  'ratio': 0.2, 'gat_layer1_out_dim': 90}
    elif args.task_name=='bt_549':
        hyperp={'dropout': 0.0, 'lr': 0.0016, 'gqmnn_hidden_dim1': 103, 'gqmnn_hidden_dim2': 956, 'gqmnn_hidden_dim3': 624, 'fpn_hidden_dim': 633, 'gat_ci_out': 102, 'gat_e_out_dim': 203, 'weight_decay': 0.001, 'fpn_dropout': 0.73, 'fpn_out_dim': 715, 'gnn_dropout': 0.35, 'nheads': 4, 'ratio': 0.8, 'gat_layer1_out_dim': 165}
    elif args.task_name == 'mcf_7':
        hyperp={'dropout': 0.5, 'lr': 0.0027, 'gqmnn_hidden_dim1': 925, 'gqmnn_hidden_dim2': 940, 'gqmnn_hidden_dim3': 700, 'fpn_hidden_dim': 432, 'gat_ci_out': 100, 'gat_e_out_dim': 200, 'weight_decay': 0.0002, 'fpn_dropout': 0.4, 'fpn_out_dim': 990, 'gnn_dropout': 0.55, 'nheads': 3, 'ratio': 0.2, 'gat_layer1_out_dim': 90}
        hyperp = {'dropout': 0.1, 'lr': 0.0018, 'gqmnn_hidden_dim1': 619, 'gqmnn_hidden_dim2': 243,
              'gqmnn_hidden_dim3': 485, 'fpn_hidden_dim': 419, 'gat_ci_out': 89, 'gat_e_out_dim': 197,
              'weight_decay': 9.11e-05, 'fpn_dropout': 0.67, 'fpn_out_dim': 442, 'gnn_dropout': 0.3, 'nheads': 2,
              'ratio': 0.19, 'gat_layer1_out_dim': 56}
    elif args.task_name=='mda_mb_468':
        # (0.8965360744321762, 0.025023285975114332, 0.9104440173447076, 0.027078545282855367) 600
        hyperp = {'dropout': 0.05, 'lr': 0.0018, 'gqmnn_hidden_dim1': 620, 'gqmnn_hidden_dim2': 240,
                  'gqmnn_hidden_dim3': 485, 'fpn_hidden_dim': 420, 'gat_ci_out': 90, 'gat_e_out_dim': 200,
                  'weight_decay': 9e-05, 'fpn_dropout': 0.65, 'fpn_out_dim': 440, 'gnn_dropout': 0.3, 'nheads': 7,
                  'ratio': 0.20, 'gat_layer1_out_dim': 55}
    elif args.task_name=='mda_mb_231':
        hyperp = {'dropout': 0.1, 'lr': 0.0018, 'gqmnn_hidden_dim1': 619, 'gqmnn_hidden_dim2': 243,
                  'gqmnn_hidden_dim3': 485, 'fpn_hidden_dim': 419, 'gat_ci_out': 89, 'gat_e_out_dim': 197,
                  'weight_decay': 9.11e-05, 'fpn_dropout': 0.67, 'fpn_out_dim': 442, 'gnn_dropout': 0.3, 'nheads': 2,
                  'ratio': 0.19, 'gat_layer1_out_dim': 56}

    for key, value in hyperp.items():
        setattr(args, key, value)
    args.max_atom=0
    return args