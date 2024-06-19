import os

import torch
from torch import nn

import Graph
import torch.nn.functional as F

class GATLayerWithRNNAndWeight(nn.Module):

    def __init__(self, args, in_features, out_features, concat=True):
        super(GATLayerWithRNNAndWeight, self).__init__()
        self.args = args
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.dropout = nn.Dropout(args.gnn_dropout)
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(out_features * 2, args.gat_e_out_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.weight_fc = nn.Linear(args.gat_bonds_input_dim, args.gat_bonds_out_dim)
        self.fc = nn.Linear(args.gat_e_out_dim + args.gat_bonds_out_dim, 1)
        self.leakyrelu = nn.LeakyReLU()
        # self.trans=SelfAttention(args,out_features,args.gat_ci_out,args.gat_ci_heads)
        # self.fc_out=nn.Linear(args.gat_ci_out*2+out_features,args.gat_layer1_out_dim)
        self.fc_out = nn.Linear(args.gat_ci_out*2  + out_features, args.gat_layer1_out_dim)
        self.fcc = nn.Linear(out_features, 1)

        self.fc_c = nn.RNN(args.max_atom, args.gat_ci_out , batch_first=True, bidirectional=True)


    def forward(self, input, adj, weight, smiles=None, no=None):
        atom_feature = torch.matmul(input, self.W)
        N = atom_feature.size()[1]
        batch_size = atom_feature.size()[0]
        atom_trans = torch.cat([atom_feature.repeat(1, 1, N).view(batch_size, N * N, -1), atom_feature.repeat(1, N, 1)],
                               dim=1).view(batch_size, N, -1, 2 * self.out_features)

        e = torch.matmul(atom_trans, self.a)

        # weight=self.dropout(weight)
        weight = self.weight_fc(weight)

        e = torch.cat([e, weight], dim=3)
        e = self.leakyrelu(e)
        e = self.fc(e).squeeze(dim=3)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=2)
        attention = self.dropout(attention)
        at_input = atom_feature.unsqueeze(2).expand(-1, -1, N, -1)
        # at_input=atom_feature.unsqueeze(1).repeat(1,N,1)
        at_input = attention.unsqueeze(3) * at_input

        hn = self.fcc(at_input)

        hn = hn.squeeze(dim=-1)
        hn = self.leakyrelu(hn)
        hn,_ = self.fc_c(hn)
        hn = self.leakyrelu(hn)




        # if smiles:  # explain
        #     root_path = self.args.gci_graph_path
        #     if not  os.path.exists(root_path):
        #         os.makedirs(root_path)
        #
        #     # hn_graph=F.softmax(hn_graph,dim=-1)
        #     for i in range(len(smiles)):
        #         Graph.draw_gci_mol(smiles[i], hn[i], os.path.join(root_path, str(no.tolist()[i]) + '.png'))
        h1 = torch.matmul(attention, atom_feature)
        h1 = torch.concat([hn, h1], dim=2)
        hn = self.fc_out(h1)
        if self.concat:
            return nn.functional.elu(hn)
        else:
            return hn


class GATLayerWithRNN(nn.Module):

    def __init__(self, args, in_features, out_features, concat=True):
        super(GATLayerWithRNN, self).__init__()
        self.args = args
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.dropout = nn.Dropout(args.gnn_dropout)

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(out_features * 2, args.gat_e_out_dim)))

        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.fc = nn.Linear(args.gat_e_out_dim, 1)

        self.leakyrelu = nn.LeakyReLU()

        self.fc_out = nn.Linear(args.gat_ci_out*2  + out_features, out_features)
        self.fcc = nn.Linear(out_features, 1)
        self.fc_c = nn.RNN(args.max_atom, args.gat_ci_out, batch_first=True, bidirectional=True)

    def forward(self, input, adj,smiles=None,no=None):
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
        hn ,_= self.fc_c(hn)
        hn = self.leakyrelu(hn)

        # hn = self.trans(at_input)
        # hn = self.leakyrelu(hn)

        h1 = torch.matmul(attention, atom_feature)
        h1 = torch.concat([hn, h1], dim=2)
        hn = self.fc_out(h1)

        return nn.functional.elu(hn)