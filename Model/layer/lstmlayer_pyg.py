import base64
import copy
import random

import math
import os
import torch_geometric.explain as explain
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import inchi
from torch import nn
import torch.nn.functional as F
from torch_geometric.explain import PGExplainer, GNNExplainer
from torch_geometric.nn import GATConv,GCNConv

import Graph
from search import serach


class GATLayerOut_single_pyg(nn.Module):
    def __init__(self,args,in_features, out_features,):
        super(GATLayerOut_single_pyg, self).__init__()
        self.gat=GATConv(in_features, out_channels=out_features, dropout=args.gnn_dropout)
        #self.gat=GATConv(in_features, out_channels=out_features, heads=1, dropout=args.gnn_dropout)
        self.args=args
    def forward(self, input, edge_index,smiles=None,no=None,label=None):
        if smiles:
        # num=mol.GetNumAtoms()
           fg = serach(smiles)
           if fg:
              print('no:', no, label)
           x = input
           y = x.clone()
           edge_index = torch.nonzero(edge_index, as_tuple=False).t().contiguous()
           edges=edge_index.clone()
        x = self.gat(input, edge_index)
        # if self.args.is_explain:
        #     model=copy.deepcopy(self.gat)
        #     model.eval()
        #     model.zero_grad()
        #
        #     config = explain.ModelConfig(mode='regression', task_level='graph', return_type='raw')
        #     #
        #     #
        #     # explainer = explain.Explainer(model, algorithm=GNNExplainer(),
        #     #                               explanation_type="model",
        #     #                               model_config=config,
        #     #                               edge_mask_type='object'
        #     #                               ,node_mask_type='object')
        #     explainer = explain.Explainer(
        #         model=model,
        #         algorithm=PGExplainer(epochs=30, lr=0.003),
        #         explanation_type='phenomenon',
        #         edge_mask_type='object',
        #         model_config=config,
        #     )
        #
        #     # 针对各种节点级别或图级别的预测进行训练:
        #     for epoch in range(30):
        #         for index in [...]:  # Indices to train against.
        #             loss = explainer.algorithm.train(epoch, model, x, edge_index,
        #                                              target=target, index=index)
        #
        #         # for epoch in range(30):
        #     #     for index in [...]:  # Indices to train against.
        #     #         loss = explainer.algorithm.train(epoch, model, x, edge_index,
        #     #                                          target=label, index=index)
        #     #         loss.backward()
        #
        #
        #     # explanation = explainer(data,edge_index=1)
        #     # exp = explainer.get_prediction(x=train_dataset.x, edge_index=train_dataset.edge_index)
        #     # print(exp)
        #
        #     explanation = explainer(x=y.detach().cuda(), edge_index=edges.detach())
        #     path='bace'
        #    # print()
        #     if not os.path.exists(path):
        #         os.makedirs(path)
        #     explanation.visualize_graph(path=os.path.join(path,f'{no}.png'))
            #print(explanation)

        return nn.functional.elu(x)

class GATLayerOut_single(nn.Module):

    def __init__(self, args,in_features, out_features):
        super(GATLayerOut_single, self).__init__()
        self.args=args
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(args.gnn_dropout)

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a =nn.Parameter(torch.zeros(size=(out_features*2, 1)))

        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.fc = nn.Linear(args.gat_e_out_dim, 1)
        self.leakyrelu = nn.LeakyReLU()
        self.fc_out = nn.Linear(out_features,out_features)
        self.fcc = nn.Linear(out_features, 1)
    def forward(self, input, adj,smiles=None,no=None):
        atom_feature = torch.matmul(input, self.W)
        N = atom_feature.size()[0]
        atom_trans = torch.cat([atom_feature.repeat(1, N).view(N * N, -1), atom_feature.repeat(N, 1)], dim=1).view(N,
                                                                                                                   -1,
                                                                                                                   2 * self.out_features)
        e = torch.matmul(atom_trans, self.a).squeeze(dim=-1)
        e = self.leakyrelu(e)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=1)
        attention = self.dropout(attention)
        h1 = torch.matmul(attention, atom_feature)
        h1=nn.functional.elu(h1)

        return h1, attention


