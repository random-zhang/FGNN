import os
from copy import deepcopy, copy

import matplotlib
import torch
from matplotlib import cm
from matplotlib.colors import ListedColormap
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from torch import nn
from torch_geometric.explain import PGExplainer

import tool
from Model.compare import gci_rnn, gci_gru, gci_lstm, gci_fc, gci_noweight
from Model.layer.lstmlayer import GATLayerOut, GATLayerAndWeight, GATLayer, GATLayerOut_single
from Model.layer.lstmlayer_pyg import GATLayerOut_single_pyg
import torch_geometric.explain as explain
atts_out=[]
class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.args = args
        self.nheads = args.nheads
        self.dropout = nn.Dropout(args.gnn_dropout)
        if args.with_gci:
            if args.with_weight:
                if args.with_rnn:
                    self.attentions1 = [
                        gci_rnn.GATLayerWithRNNAndWeight(args, args.gat_atom_input_dim, args.gat_layer1_out_dim,
                                                         concat=True)
                        for _
                        in
                        range(self.nheads)]
                    self.attentions2 = [
                        gci_rnn.GATLayerWithRNN(args, args.gat_layer1_out_dim * args.nheads, args.gat_layer1_out_dim * args.nheads,
                                          concat=False).to(args.device) for _ in range(args.k_hop - 2)]

                elif args.with_fc:
                    self.attentions1 = [
                        gci_fc.GATLayerWithRNNAndWeight(args, args.gat_atom_input_dim, args.gat_layer1_out_dim,
                                                         concat=True)
                        for _
                        in
                        range(self.nheads)]
                    self.attentions2 = [
                        gci_fc.GATLayerWithRNN(args, args.gat_layer1_out_dim * args.nheads,
                                                args.gat_layer1_out_dim * args.nheads,
                                                concat=False).to(args.device) for _ in range(args.k_hop - 2)]

            else:
                self.attentions1 = [
                    gci_noweight.GATLayerWithRNNNow(args, args.gat_atom_input_dim, args.gat_layer1_out_dim, concat=True)
                    for _
                    in
                    range(self.nheads)]
                self.attentions2 = [
                    gci_noweight.GATLayerWithRNN(args, args.gat_layer1_out_dim * args.nheads,
                                            args.gat_layer1_out_dim * args.nheads,
                                            concat=False).to(args.device) for _ in range(args.k_hop - 2)]
        else:
            if args.with_weight:
                self.attentions1 = [
                    GATLayerAndWeight(args, args.gat_atom_input_dim, args.gat_layer1_out_dim, concat=True)
                    for _
                    in
                    range(self.nheads)]
            else:
                self.attentions1 = [
                    GATLayer(args, args.gat_atom_input_dim, args.gat_layer1_out_dim, concat=True)
                    for _
                    in
                    range(self.nheads)]
            self.attentions2 = [
                GATLayer(args, args.gat_layer1_out_dim * args.nheads,
                                             args.gat_layer1_out_dim * args.nheads,
                                             concat=False).to(args.device) for _ in range(args.k_hop - 2)]
        if args.use_pyg:
            self.out_at = GATLayerOut_single_pyg(args, args.gat_layer1_out_dim * args.nheads,
                                                 int(args.fpn_out_dim * 2 * args.ratio)).to(args.device)

        else:
            self.out_at = GATLayerOut_single(args, args.gat_layer1_out_dim * args.nheads,
                                             int(args.fpn_out_dim * 2 * args.ratio)).to(args.device)


        #self.out_at=SparseGATLayer( args.gat_layer1_out_dim * args.nheads,int(args.fpn_out_dim * 2 * args.ratio)).to(args.device)
        # tool.setSeed(args.seed)
        #self.out_at=GATConv(args.gat_layer1_out_dim * args.nheads,int(args.fpn_out_dim * 2 * args.ratio),heads=1,dropout=args.gnn_dropout).to(args.device)
        for i, attention in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention)
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention)
        self.add_module('out_at', self.out_at)
        self.elu = nn.ELU()
        self.fc=nn.Linear(args.max_atom,1)


    def forward(self, atom_feature,weight,adj,smiles=None,no=None,labels=None):  # smiles

        mole_out = self.dropout(atom_feature)
        mole_out = torch.cat([att(mole_out, adj, weight,smiles,no) for att in self.attentions1], dim=2,)
        mole_out = self.dropout(mole_out)


        for atts in self.attentions2:
            res = mole_out
            mole_out = atts(mole_out, adj)
            mole_out = self.elu(mole_out)
            mole_out = self.dropout(mole_out)
            mole_out += res






        if self.args.use_pyg:
            out = []
            for i in range(len(mole_out)):
                edge_index = tool.adj_to_sparse(adj[i]).coalesce()
                # out.append(self.out_at(mole_out[i], edge_index))
                # p=self.out_at(mole_out[i], adj[i], smiles[i], no[i],labels[i])
                p = self.out_at(mole_out[i], edge_index)
                out.append(p)
            out = torch.stack(out, dim=0)
            if self.args.is_explain:
                config = explain.ModelConfig(mode='regression', task_level='graph', return_type='raw')
                #
                #
                # explainer = explain.Explainer(model, algorithm=GNNExplainer(),
                #                               explanation_type="model",
                #                               model_config=config,
                #                               edge_mask_type='object'
                #                               ,node_mask_type='object')
                model = deepcopy(self.out_at)
                # explainer = explain.Explainer(
                #     model=self.out_at,
                #     algorithm=PGExplainer(epochs=100, lr=0.003).cuda(),
                #     explanation_type='phenomenon',
                #     edge_mask_type='object',
                #
                #
                #     model_config=config,
                # )
                explainer = explain.Explainer(
                    model=self.out_at,
                    algorithm=PGExplainer(epochs=30, lr=0.0001).cuda(),
                    explanation_type='phenomenon',
                    edge_mask_type='object',

                    model_config=config,
                )
                # 针对各种节点级别或图级别的预测进行训练:
                for epoch in range(30):

                    for index in range(len(mole_out)):  # Indices to train against.

                        edge_index = torch.nonzero(adj[index], as_tuple=False).t().contiguous()
                        # loss = explainer.algorithm.train(epoch, self.out_at, mole_out[index], edge_index,
                        #
                        #                                  target=None, index=index)

                        x = deepcopy(mole_out[index].detach())
                        edge_index = deepcopy(edge_index.detach().to(self.args.device))
                        target = deepcopy(out[index].detach())
                        loss = explainer.algorithm.train(epoch, model, x, edge_index, target=target)
                       # print(loss)
                        # print(loss)
                mols=deepcopy(mole_out.detach())
                target=deepcopy(out.detach())
                for i in range(len(mols)):
                    try:
                        if serach(smiles[i]):
                            print(no[i])
                        else:
                            continue
                        edge_index = torch.nonzero(adj[i], as_tuple=False).t().contiguous()
                        if not  os.path.exists(self.args.task_name):
                           os.makedirs(self.args.task_name)
                        path = os.path.join(self.args.task_name, f'{no[i]}.png')


                        explanation = explainer(mols[i], edge_index, target=target[i], index=0)
                        explanation.visualize_graph(path)
                        mask = explanation.stores[0]['edge_mask']
                        mol = Chem.MolFromSmiles(smiles[i])

                        min_value = mask.min()
                        max_value = mask.max()
                        norm = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
                        cmap = cm.get_cmap('Oranges')
                        custom_colors = ['#FFFFFF','#FACCFF','#DCB0FF','#BE93FD','#A178DF','#845EC2','#250B65']
                        bond_colors = {}
                        bond_list = []
                        # 创建自定义颜色映射
                        custom_cmap = ListedColormap(custom_colors)
                        cmap = cm.get_cmap('plasma')
                        cmap = cm.get_cmap('viridis')
                        cmap = cm.get_cmap('cool')
                        #plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap.reversed())
                        plt_colors = cm.ScalarMappable(norm=norm, cmap=custom_cmap)
                        for j in range(edge_index.shape[1]):
                            a, b = int(edge_index[0, j].cpu()), int(edge_index[1, j].cpu())
                            bond = mol.GetBondBetweenAtoms(a, b).GetIdx()
                            bond_list.append(bond)
                            bond_color = plt_colors.to_rgba(mask[j].cpu())
                            bond_colors[bond] = bond_color
                        drawer = rdMolDraw2D.MolDraw2DCairo(300, 150)
                        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol,
                                                           highlightBonds=bond_list,
                                                           highlightBondColors=bond_colors,
                                                           )

                        path = os.path.join(self.args.task_name, f'{no[i]}_1.png')
                        with open(path, 'wb') as file:
                            file.write(drawer.GetDrawingText())
                        print('writer')
                    except Exception:
                        print(Exception)
                        continue
        else:
            out = []

            for i in range(len(mole_out)):

                # out.append(self.out_at(mole_out[i], edge_index))
                # p=self.out_at(mole_out[i], adj[i], smiles[i], no[i],labels[i])
                p = self.out_at(mole_out[i], adj[i])
                out.append(p)
            out = torch.stack(out, dim=0)

        out = out.permute(0, 2, 1)
        out = self.fc(out).squeeze(dim=-1)
                # for i in range()
                # bond = mol.GetBondBetweenAtoms(r, l).GetIdx()
                # bond_list.append(bond)
                # bond_colors[bond] = bond_color
                # a=explanation.get_masked_prediction((mole_out[i],  torch.nonzero(adj[i], as_tuple=False).t().contiguous()))
                # print(a)
        return out

