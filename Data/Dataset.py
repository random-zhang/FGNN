from torch.utils.data import Dataset ,DataLoader
class MixDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def label(self):
        return [d.label[0] for d in self.data]
    def __getitem__(self, index):
        fp=self.data[index].fp
        atom_feature=self.data[index].atom_features
        weight=self.data[index].weight
        adj=self.data[index].adj
        labels=self.data[index].label
        smile=self.data[index].smile
        no=self.data[index].no
        return atom_feature,weight,adj,fp,labels,smile,no