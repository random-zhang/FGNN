import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from rdkit import Chem
import matplotlib.cm as cm
from rdkit.Chem import inchi, Descriptors, Crippen
from rdkit.Chem.Draw import rdMolDraw2D

def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

def draw_mol(smile, smi_att,path,adj):
    smi_att = np.array(smi_att.cpu().detach())
    mol = Chem.MolFromSmiles(smile)

    atom_num=mol.GetNumAtoms()
    smi_att=smi_att[:atom_num,:atom_num]
    adj = np.array(adj.cpu().detach())
    adj=adj[:atom_num,:atom_num]
    smi_att=smi_att*adj

    for i in range(atom_num):
        for j in range(i + 1):
           # smi_att[j][i] = abs(smi_att[j][i]) + abs(smi_att[i][j])
           #  smi_att[j][i] = abs(smi_att[i][j])
            smi_att[j][i] = abs(smi_att[i][j])
            smi_att[i][j] = 0

    min_value = smi_att.min(axis=(0, 1))
    max_value = smi_att.max(axis=(0, 1))
    norm = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value )
    cmap = cm.get_cmap('Oranges')
    # custom_colors = ['#00FF00','#00CC00','#009900','#006600','#003300']

    # 创建自定义颜色映射
    #custom_cmap = ListedColormap(custom_colors)
    cmap = cm.get_cmap('plasma')
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap.reversed())


    mol = mol_with_atom_index(mol)

    bond_list = []
    bond_colors = {}
    bond_no = np.nonzero(smi_att)

    for i in range(len(bond_no[0])):
        r = int(bond_no[0][i])
        l = int(bond_no[1][i])

        bond_color = smi_att[r, l]
        bond_color = plt_colors.to_rgba(bond_color)
        if  mol.GetBondBetweenAtoms(r, l):
           bond = mol.GetBondBetweenAtoms(r, l).GetIdx()
           bond_list.append(bond)
           bond_colors[bond] = bond_color


    drawer = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol,
                                       highlightBonds=bond_list,
                                       highlightBondColors=bond_colors,
                                       )



    with open(path, 'wb')as file:
        file.write(drawer.GetDrawingText())






def draw_mola(smile, h, path):

    mol = Chem.MolFromSmiles(smile)

    atom_num=mol.GetNumAtoms()
    smi_att=h[:atom_num,]



    logp_values=np.array([0.34, 0.34, 0.34, 0.34, -0.10, -0.10, -0.05, 0.07, -0.07, -0.08, 0.86, 0.16, 0.34, 0.34, 0.34, 0.34, 0.34, -0.35, -0.17, -0.17, 0.40, -0.08, -0.08]
)
    v=np.corrcoef(smi_att, logp_values)[0, 1]
    print('correlation between:',v)
    min_value = smi_att.min()
    max_value = smi_att.max()
    norm = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)

    cmap = cm.get_cmap('plasma')
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap.reversed())


    mol = mol_with_atom_index(mol)
    atom_c={}
    for i in range(smi_att.shape[0]):
        atom_c[i]=plt_colors.to_rgba(smi_att[i])
    drawer = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol,

                                       highlightAtoms=list(atom_c.keys()),
                                       highlightAtomColors=atom_c
                                       )

    drawer.FinishDrawing()

    with open(path, 'wb')as file:
        file.write(drawer.GetDrawingText())
def draw_gci_mol(smile, smi_att, path):
    smi_att = np.array(smi_att.cpu().detach())
    mol = Chem.MolFromSmiles(smile)
    atom_num = mol.GetNumAtoms()
    smi_att = smi_att[:atom_num]
    smi_att=smi_att.mean(-1)
    cmap = cm.get_cmap('Oranges')
    plt_colors = cm.ScalarMappable( cmap=cmap)
    mol = mol_with_atom_index(mol)
    atom_colors = {}
    for i in range(len(smi_att)):
        atom_colors[i]=plt_colors.to_rgba(smi_att[i])
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol,
                                       highlightAtoms=list(atom_colors.keys()),
                                       highlightAtomColors=atom_colors)

    with open(path, 'wb') as file:
        file.write(drawer.GetDrawingText())
