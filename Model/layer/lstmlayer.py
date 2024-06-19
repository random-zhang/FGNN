import base64
import random

import math
import os

import numpy as np
import torch
from rdkit.Chem import inchi
from torch import nn


import Graph


class GATLayerWithOrderAndWeight(nn.Module):

    def __init__(self, args,in_features, out_features, concat=True):
        super(GATLayerWithOrderAndWeight, self).__init__()
        self.args=args
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.dropout = nn.Dropout(args.gnn_dropout)
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a =nn.Parameter(torch.zeros(size=(out_features*2, args.gat_e_out_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.weight_fc=nn.Linear(args.gat_bonds_input_dim,args.gat_bonds_out_dim)
        self.fc=nn.Linear(args.gat_e_out_dim+args.gat_bonds_out_dim,1)
        self.leakyrelu = nn.LeakyReLU()
        #self.trans=SelfAttention(args,out_features,args.gat_ci_out,args.gat_ci_heads)
       # self.fc_out=nn.Linear(args.gat_ci_out*2+out_features,args.gat_layer1_out_dim)
        self.fc_out = nn.Linear( args.gat_ci_out*2+out_features, args.gat_layer1_out_dim)
        self.fcc=nn.Linear(out_features,1)
        self.fc_c = nn.LSTM(args.max_atom, args.gat_ci_out, batch_first=True, bidirectional=True)

    def forward(self, input, adj,weight):
        atom_feature = torch.matmul(input, self.W)
        N = atom_feature.size()[1]
        batch_size=atom_feature.size()[0]
        atom_trans = torch.cat([atom_feature.repeat(1,1, N).view(batch_size,N * N, -1), atom_feature.repeat(1,N, 1)], dim=1).view(batch_size,N, -1,2 * self.out_features )

        e=torch.matmul(atom_trans, self.a)

        # weight=self.dropout(weight)
        weight=self.weight_fc(weight)

        e = torch.cat([e,weight ], dim=3)
        e = self.leakyrelu(e)
        e=self.fc(e).squeeze(dim=3)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=2)
        attention = self.dropout(attention)
        at_input= atom_feature.unsqueeze(2).expand(-1, -1, N, -1)
        #at_input=atom_feature.unsqueeze(1).repeat(1,N,1)
        at_input=attention.unsqueeze(3)*at_input

        hn=self.fcc(at_input)
        hn=hn.squeeze(dim=-1)
        hn=self.leakyrelu(hn)
        hn,_=self.fc_c(hn)
        hn = self.leakyrelu(hn)

        # hn = self.trans(at_input)
        # hn = self.leakyrelu(hn)
        h1 = torch.matmul(attention, atom_feature)
        h1 = torch.concat([hn, h1], dim=2)
        hn = self.fc_out(h1)
        if self.concat:
            return nn.functional.elu(hn)
        else:
            return hn

class GATLayerWithOrder(nn.Module):

    def __init__(self, args,in_features, out_features, concat=True):
        super(GATLayerWithOrder, self).__init__()
        self.args=args
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.dropout = nn.Dropout(args.gnn_dropout)

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a =nn.Parameter(torch.zeros(size=(out_features*2, args.gat_e_out_dim)))

        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.fc = nn.Linear(args.gat_e_out_dim, 1)

        self.leakyrelu = nn.LeakyReLU()
        #self.trans = SelfAttention(args,out_features, args.gat_ci_out, args.gat_ci_heads)
        #self.fc_out = nn.Linear( args.gat_ci_out*2 + out_features, out_features)
        #self.fc_out = nn.Linear( out_features, out_features)
        self.fc_out = nn.Linear(args.gat_ci_out *2  + out_features,out_features)
        self.fcc = nn.Linear(out_features, 1)
        self.fc_c = nn.LSTM(args.max_atom,  args.gat_ci_out, batch_first=True, bidirectional=True)
    def forward(self, input, adj):
        atom_feature = torch.matmul(input, self.W)
        N = atom_feature.size()[1]
        batch_size = atom_feature.size()[0]
        atom_trans = torch.cat([atom_feature.repeat(1, 1, N).view(batch_size, N * N, -1), atom_feature.repeat(1, N, 1)],
                               dim=1).view(batch_size, N, -1, 2 * self.out_features)

        e = torch.matmul(atom_trans, self.a)


        e = self.leakyrelu(e)
        e = self.fc(e).squeeze(dim=3)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=2)
        attention = self.dropout(attention)
        at_input = atom_feature.unsqueeze(2).expand(-1, -1, N, -1)

        at_input = attention.unsqueeze(3) * at_input

        hn = self.fcc(at_input)
        hn = hn.squeeze(dim=-1)
        hn = self.leakyrelu(hn)
        hn, _ = self.fc_c(hn)
        hn = self.leakyrelu(hn)

        # hn = self.trans(at_input)
        # hn = self.leakyrelu(hn)

        h1 = torch.matmul(attention, atom_feature)
        h1 = torch.concat([hn, h1], dim=2)
        hn = self.fc_out(h1)

        return nn.functional.elu(hn)

class GATLayerAndWeight(nn.Module):

    def __init__(self, args,in_features, out_features, concat=True):
        super(GATLayerAndWeight, self).__init__()
        self.args=args
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.dropout = nn.Dropout(args.gnn_dropout)
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a =nn.Parameter(torch.zeros(size=(out_features*2, args.gat_e_out_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.weight_fc=nn.Linear(args.gat_bonds_input_dim,args.gat_bonds_out_dim)
        self.fc=nn.Linear(args.gat_e_out_dim+args.gat_bonds_out_dim,1)
        self.leakyrelu = nn.LeakyReLU()
        self.fc_out = nn.Linear(out_features, args.gat_layer1_out_dim)


    def forward(self, input, adj,weight, smiles=None, no=None):
        atom_feature = torch.matmul(input, self.W)
        N = atom_feature.size()[1]
        batch_size=atom_feature.size()[0]
        atom_trans = torch.cat([atom_feature.repeat(1,1, N).view(batch_size,N * N, -1), atom_feature.repeat(1,N, 1)], dim=1).view(batch_size,N, -1,2 * self.out_features )

        e=torch.matmul(atom_trans, self.a)

        weight=self.weight_fc(weight)

        e = torch.cat([e,weight ], dim=3)
        e = self.leakyrelu(e)
        e=self.fc(e).squeeze(dim=3)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=2)
        attention = self.dropout(attention)
        h1 = torch.matmul(attention, atom_feature)
        hn = self.fc_out(h1)
        if self.concat:
            return nn.functional.elu(hn)
        else:
            return hn

class GATLayer(nn.Module):

    def __init__(self, args,in_features, out_features, concat=True):
        super(GATLayer, self).__init__()
        self.args=args
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.dropout = nn.Dropout(args.gnn_dropout)

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a =nn.Parameter(torch.zeros(size=(out_features*2, args.gat_e_out_dim)))

        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.fc = nn.Linear(args.gat_e_out_dim, 1)

        self.leakyrelu = nn.LeakyReLU()

        self.fc_out = nn.Linear(out_features,out_features)


    def forward(self, input, adj, weight=None,smiles=None, no=None):
        atom_feature = torch.matmul(input, self.W)
        N = atom_feature.size()[1]
        batch_size = atom_feature.size()[0]
        atom_trans = torch.cat([atom_feature.repeat(1, 1, N).view(batch_size, N * N, -1), atom_feature.repeat(1, N, 1)],
                               dim=1).view(batch_size, N, -1, 2 * self.out_features)
        e = torch.matmul(atom_trans, self.a)
        e = self.leakyrelu(e)
        e = self.fc(e).squeeze(dim=3)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=2)
        attention = self.dropout(attention)
        h1 = torch.matmul(attention, atom_feature)
        hn = self.fc_out(h1)
        if self.concat:
            return nn.functional.elu(hn)
        else:
            return hn





class GATLayerOut(nn.Module):

    def __init__(self, args,in_features, out_features):
        super(GATLayerOut, self).__init__()
        self.args=args
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(args.gnn_dropout)

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a =nn.Parameter(torch.zeros(size=(out_features*2, 1)))

        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.fc = nn.Linear(args.gat_e_out_dim, 1)
        # self.leakyrelu = nn.LeakyReLU(args.leakyrelu_aplha)\
        self.leakyrelu = nn.LeakyReLU()
        self.fc_out = nn.Linear(out_features,out_features)
        self.fcc = nn.Linear(out_features, 1)
    def forward(self, input, adj,smiles=None,no=None):
        atom_feature = torch.matmul(input, self.W)
        N = atom_feature.size()[1]
        batch_size = atom_feature.size()[0]

        atom_trans = torch.cat([atom_feature.repeat(1, 1, N).view(batch_size, N * N, -1), atom_feature.repeat(1, N, 1)],
                               dim=1).view(batch_size, N, -1, 2 * self.out_features)

        e = torch.matmul(atom_trans, self.a)
        e = self.leakyrelu(e).squeeze(dim=-1)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=2)

        if smiles:#explain
            root_path = self.args.graph_path
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            for i in range(len(smiles)):

                Graph.draw_mol(smiles[i],attention[i],os.path.join(root_path,str(no.tolist()[i])+'.png'))
        attention = self.dropout(attention)
        h1 = torch.matmul(attention, atom_feature)

        return nn.functional.elu(h1)

import torch.sparse as sp

#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_sparse import spmm
# from torch_scatter import scatter_softmax
#
# class SparseGATLayer(nn.Module):
#
#
#     def __init__(self,input_dim, out_dim, concat=True):
#         super(SparseGATLayer, self).__init__()
#         self.in_features = input_dim
#         self.out_features = out_dim
#
#         self.concat = concat
#
#         self.W = nn.Parameter(torch.zeros(size=(input_dim, out_dim)))
#         self.attn = nn.Parameter(torch.zeros(size=(1, 2 * out_dim)))
#         nn.init.xavier_normal_(self.W, gain=1.414)
#         nn.init.xavier_normal_(self.attn, gain=1.414)
#         # nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         # nn.init.xavier_uniform_(self.attn.data, gain=1.414)
#         self.leakyrelu = nn.LeakyReLU()
#     def forward(self,x, edge):
#         '''
#         :param x:   dense tensor. size: nodes*feature_dim
#         :param adj:    parse tensor. size: nodes*nodes
#         :return:  hidden features
#         '''
#
#         N = x.size()[0]   # 图中节点数
#
#         if x.is_sparse:   # 判断特征是否为稀疏矩阵
#             h = torch.sparse.mm(x, self.W)
#         else:
#             h = torch.mm(x, self.W)
#         # Self-attention (because including self edges) on the nodes - Shared attention mechanism
#         edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()  # edge_h: 2*D x E
#         values = self.attn.mm(edge_h).squeeze()   # 使用注意力参数对特征进行投射
#         edge_e_a = self.leakyrelu(values)  # edge_e_a: E   attetion score for each edge，对应原论文中的添加leakyrelu操作
#         # 由于torch_sparse 不存在softmax算子，所以得手动编写，首先是exp(each-max),得到分子
#         edge_e = torch.exp(edge_e_a - torch.max(edge_e_a))
#
#         # 使用稀疏矩阵和列单位向量的乘法来模拟row sum，就是N*N矩阵乘N*1的单位矩阵的到了N*1的矩阵，相当于对每一行的值求和
#         e_rowsum = spmm(edge, edge_e, m=N, n=N, matrix=torch.ones(size=(N, 1)).cuda())  # e_rowsum: N x 1，spmm是稀疏矩阵和非稀疏矩阵的乘法操作
#         h_prime = spmm(edge, edge_e, n=N,m=N, matrix=h)   # 把注意力评分与每个节点对应的特征相乘
#
#         h_prime = h_prime.div(e_rowsum + torch.Tensor([9e-15]).cuda())  # h_prime: N x out，div一看就是除，并且每一行的和要加一个9e-15防止除数为0
#         # softmax结束
#         if self.concat:
#             # if this layer is not last layer
#             return F.elu(h_prime)
#         else:
#             # if this layer is last layer
#             return h_prime
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
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
    def forward(self, input, adj,smiles=None,no=None,label=None):

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

        # if self.args.graph_explain:

        attention = self.dropout(attention)
        h1 = torch.matmul(attention, atom_feature)
        h1=nn.functional.elu(h1)
        # if smiles:  # explain
        #     root_path = self.args.graph_path
        #     if not os.path.exists(root_path):
        #         os.makedirs(root_path)
        #     # for i in range(len(smiles)):
        #     # Graph.draw_mol(smiles, attention, os.path.join(root_path, str(no) + '.png'))
        #     # smiles = smiles.replace('\\', '/')
        #     h=h1.cpu().detach().numpy().mean(axis=1)
        #     no=str(int(no.cpu().detach()))
        #
        #     # if no=='126':
        #     Graph.draw_mol(smiles, attention, os.path.join(root_path, no + '.png'), adj)

            #Graph.draw_mola(smiles, h, os.path.join(root_path, no + '.png'))
        return h1


# seed=100
# random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
#
# m=SparseGATLayer(32,64).cuda()
# x = torch.randn(20, 32).cuda()  # 输入张量
# edge_index=torch.randint(2,(2,10)).cuda()
# a=m(x,edge_index)
# print(a)
# a=m(x,edge_index)
# print(a)a=m(x,edge_index)
# print(a)