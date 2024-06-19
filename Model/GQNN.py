import argparse

from shap import DeepExplainer

from Model.GAT import *
from Model.FPN import *
from Model.DN import *



class GQModel(nn.Module):
    def __init__(self, args):
        super(GQModel, self).__init__()
        self.args = args
        self.sigmoid = nn.Sigmoid()
        if args.fp_dim > 0:
            in_dim = args.fpn_out_dim * 2  # + args.dn_out_dimZ
        else:
            in_dim = int(args.fpn_out_dim * 2 * args.ratio)
        if self.args.ratio != 0:
            self.encoder1 = GAT(args)

        self.out = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(in_features=in_dim, out_features=args.gqmnn_hidden_dim1, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(in_features=args.gqmnn_hidden_dim1, out_features=args.gqmnn_hidden_dim2, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(in_features=args.gqmnn_hidden_dim2, out_features=args.gqmnn_hidden_dim3, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout),
            nn .Linear(in_features=args.gqmnn_hidden_dim3, out_features=len(args.labels), bias=True)
        )
        if self.args.ratio != 1:
            self.encoder2 = FPN(args)

    def forward(self, atom_feature, weight, adj, fp,smiles=None,no=None,labels=None):
        if self.args.ratio == 0:
            output = self.encoder2(fp)
        elif self.args.ratio == 1:
            output = self.encoder1(atom_feature, weight, adj, smiles,no,labels)
        else:
            output = self.encoder1(atom_feature, weight, adj, smiles,no,labels)


            if self.args.fp_dim > 0:
                fpn_out = self.encoder2(fp)
            output = torch.cat([output, fpn_out], dim=1)  # 拼接

        output = self.out(output)

        if not self.training and self.args.dataset_type == 'classification':
            output = self.sigmoid(output)

        return output