

class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(len(s))))
            self.dim += len(s)+1

    def encode(self, inputs):
        output=[]
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            out=[0]*(len(feature_mapping)+1)
            if feature not in feature_mapping:
                out[-1]=1
            else:
                out[feature_mapping[feature]] = 1
            output.extend(out)

        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()
    def degree(self, atom):
        return atom.GetTotalDegree()
    def formal_charge(self, atom):
        return atom.GetFormalCharge()
    def charity_type(self, atom):
        return int(atom.GetChiralTag())

class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def encode(self, bond):


        if bond is None:
            output=[0]*self.dim
            output[1] = 1.0
            return output
        else:

            output = super().encode(bond)
            return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()
    def is_aromatic(self, bond):

        return  bond.GetIsAromatic()


atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": {"B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"},
        'degree': [0, 1, 2, 3, 4, 5],
        'formal_charge': [-1, -2, 1, 2, 0],
        'charity_type': [0, 1, 2, 3],
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3"},

    }
)



bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
        "is_aromatic": {True, False},
    }
)
