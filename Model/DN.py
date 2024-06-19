from torch import nn
from rdkit import Chem
import torch
from rdkit.Chem import Descriptors
def get_aromatic_rings(mol):
    rings = Chem.GetSymmSSSR(mol)
    aromatic_rings = []

    for ring in rings:
        is_aromatic = True
        for atom_idx in ring:
            atom = mol.GetAtomWithIdx(atom_idx)
            if not atom.GetIsAromatic():
                is_aromatic = False
                break
        if is_aromatic:
            aromatic_rings.append(ring)

    return aromatic_rings
class DN(nn.Module):
    def __init__(self,args):
       super(DN,self).__init__()
       self.args=args
       self.bn=nn.Sequential(
           nn.ELU(),
           nn.Linear(8, args.dn_out_dim),

       )
    def forward(self,mols):
        D=[]
        for one in mols:
            feature=[]
            mol = one.mol
            aromatic_rings = get_aromatic_rings(mol)
            feature.append(len(aromatic_rings))
            formal_charge = Chem.GetFormalCharge(mol)
            feature.append(formal_charge)
            max_conjugate = 0
            atom_indices = []
            for ring in aromatic_rings:
                atoms = [mol.GetAtomWithIdx(index) for index in ring]
                atoms_index = [atom.GetIdx() for atom in atoms]
                atom_indices.append(set(atoms_index))
            for i in range(len(atom_indices)):
                sum = 0
                for j in range(len(atom_indices)):
                    if i != j:
                        common = atom_indices[i].intersection(atom_indices[j])
                        if len(common) >= 2:
                            sum += 1
                if sum > max_conjugate:
                    max_conjugate = sum
            #feature.append(max_conjugate/10)
            feature.append(max_conjugate)
            mol_weight = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            bonds = mol.GetBonds()

            # 统计可旋转键数目
            num_rotatable_bonds = 0
            for bond in bonds:
                if bond.GetBeginAtom().IsInRing() or bond.GetEndAtom().IsInRing():
                    continue
                if not bond.GetBeginAtom().IsInRingSize(3) and not bond.GetEndAtom().IsInRingSize(3):
                    num_rotatable_bonds += 1

            feature.append(logp)

            feature.append(hba)
            feature.append(hbd)
            feature.append(mol_weight)
            feature.append(num_rotatable_bonds)
            D.append(feature)
        features=torch.Tensor(D).to(self.args.device)
        features=self.bn(features)
        return features