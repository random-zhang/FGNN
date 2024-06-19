
import torch

from torch import nn


class FPN(nn.Module):
    def __init__(self, args):
        super(FPN, self).__init__()
        self.dropout_fpn =args.fpn_dropout
        self.args = args
        if hasattr(args, 'fp_type'):
            self.fp_type = args.fp_type
        else:
            self.fp_type = 'mixed'
        self.fp_dim = args.fp_dim#2513

        self.layer1=nn.Sequential(
            nn.Dropout(args.fpn_dropout),

            nn.Linear(self.fp_dim, args.fpn_hidden_dim),
        )
        self.leakyrelu= nn.LeakyReLU()
        self.bn=nn.BatchNorm1d(args.fp_dim)
        self.fc = nn.Linear(args.fpn_hidden_dim,args.fpn_out_dim*2-int(args.fpn_out_dim * 2 * args.ratio))
    def forward(self, fps):
        fpn_out = self.layer1(fps)
        fpn_out=self.leakyrelu(fpn_out)
        fpn_out=self.fc(fpn_out)
        return fpn_out
