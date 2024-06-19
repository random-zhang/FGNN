# Import libararies
import numpy as np
import pandas as pd
import os
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
import torch_geometric
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
import urllib.request
import tarfile
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

# Load the dataset
dataset = TUDataset(root='data/TUDataset', name='MUTAG')
# print details about the graph
print(f'Dataset: {dataset}:')
print("Number of Graphs: ", len(dataset))
print("Number of Freatures: ", dataset.num_features)
print("Number of Classes: ", dataset.num_classes)
data = dataset[0]
print(data)
print("No. of nodes: ", data.num_nodes)
print("No. of Edges: ", data.num_edges)
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
# Create train and test dataset
torch.manual_seed(12345)
dataset = dataset.shuffle()
train_dataset = dataset[:50]
test_dataset = dataset[50:]
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')
'''graphs in graph classification datasets are usually small, 
a good idea is to batch the graphs before inputting 
them into a Graph Neural Network to guarantee full GPU utilization__
_In pytorch Geometric adjacency matrices are stacked in a diagonal fashion 
(creating a giant graph that holds multiple isolated subgraphs), a
nd node and target features are simply concatenated in the node dimension:
'''
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()


# Build the model
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


model = GNN(hidden_channels=64).cuda()
print(model)
# set the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
# set the loss function
criterion = torch.nn.CrossEntropyLoss()


# Creating the function to train the model
def train():
    model.train()
    for data in train_loader:
        # Iterate in batches over the training dataset.
        out = model(data.x.cuda(), data.edge_index.cuda(), data.batch.cuda())  # Perform a single forward pass.
        loss = criterion(out, data.y.cuda())  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

import torch_geometric.explain as explain
# function to test the model
def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.cuda(), data.edge_index.cuda(), data.batch.cuda())
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y.cuda()).sum())  # Check against ground-truth labels.

        config = explain.ModelConfig(mode='binary_classification', task_level='graph', return_type='raw')
        explainer = explain.Explainer(model, algorithm=explain.GNNExplainer(), explanation_type="model",
                                      model_config=config,
                                      node_mask_type='object', edge_mask_type='object')
        # explanation = explainer(data,edge_index=1)
        # exp = explainer.get_prediction(x=train_dataset.x, edge_index=train_dataset.edge_index)
        # print(exp)

        explanation = explainer(x=data.x.cuda(), edge_index=data.edge_index.cuda(),batch=data.batch.cuda())
        explanation.visualize_graph(path='1.png')
        print(explanation)

        # 查看解释

        # 查看解释
       # print(exp)
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


# Train the model for 150 epochs
for epoch in range(1, 160):
    train()
    #train_acc = test(train_loader)
    print('epoch {}'.format(epoch))
    test_acc = test(test_loader)
    if (epoch % 10 == 0):
        '''print(f'Epoch {epoch:>3} | Train Loss: {total_loss/len(train_loader):.3f} '
              f'| Train Acc: {acc/len(train_loader)*100:>6.2f}% | Val Loss: '
              f'{val_loss/len(train_loader):.2f} | Val Acc: '
              f'{val_acc/len(train_loader)*100:.2f}%')
        '''
        print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}')
# Explain the Graph



# explainer = GNNExplainer(model, epochs=100,return_type='log_prob')
# data = dataset[0]
# node_feat_mask, edge_mask = explainer.explain_graph(data.x, data.edge_index)
# ax, G = explainer.visualize_subgraph(-1,data.edge_index, edge_mask, data.y)
# plt.show()