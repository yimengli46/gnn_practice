import numpy as np 
from torch_geometric.data import Data 
import torch

num_nodes = 5

nodes = np.random.rand(num_nodes, 1)

edge_index = np.reshape(np.meshgrid(np.arange(num_nodes), np.arange(num_nodes)), (2,-1))
# edge_index = torch.tensor([[0,1], [1,2]])
x = torch.tensor(nodes)
# x = torch.tensor([[1], [2], [3]])
edge_index = torch.tensor(edge_index, dtype=torch.long)
y = torch.zeros(num_nodes, dtype=torch.bool)
y[nodes.argmin(0)] = True

print(x)
print(y)
print(edge_index)
data = Data(x=x, edge_index=edge_index, y=y)

import newtorkx as nx 
import matplotlib.pyplot as plt 
from torch_geometric.utils import to_networkx

def visualize(h, color, epoch=None, loss=None):
	plt.figure(figsize=(7, 7))
	plt.xticks([])
	plt.yticks([])

	if torch.is_tensor(h):
		h = h.detach().cpu().numpy()
		plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap='Set2')
		if epoch is not None and loss is not None:
			plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
		else:
			nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True, node_color=color, cmap='Set2')
	plt.show()

#G = to_networkx(data, to_undirected=True)
#visualize(G, color=data.y)

import torch_geometric.utils as utils
import networkx as nx
graph = utils.to_networkx(data)

def random_graph(num_nodes=None):
	if num_nodes is None:
		num_nodes = np.random.randint(5, 10)

	nodes = np.random.rand(num_nodes, 1)

	edge_index = np.reshape(np.meshgrid(np.arange(num_nodes), np.arange(num_nodes)), (2,-1))
	x = torch.tensor(nodes, dtype=torch.double)
	edge_index = torch.tensor(edge_index, dtype=torch.long)
	y = torch.zeros(num_nodes, dtype=torch.long)
	y[nodes.argmin(0)] = True
	data = Data(x=x, edge_index=edge_index, y=y)
	return data

data = random_graph()
#G = to_networkx(data, to_undirected=True)
#visualize(G, color=data.y)

from torch_geometric.data import DataLoader

size = 50
dataset = [random_graph() for i in range(size)]
dataloader = DataLoader(dataset, batch_size=16)

from torch.nn import Linear
from torch_geometric.nn import GCNConv

NUM_FEATURES = 1
NUM_CLASSES = 1

class GCN(torch.nn.Module):
	def __init__(self):
		super(GCN, self).__init__()
		torch.manual_seed(12345)
		self.conv1 = GCNConv(NUM_FEATURES, 4)
		self.conv2 = GCNConv(4, 4)
		self.conv3 = GCNConv(4, 2)
		self.classifier = Linear(2, NUM_CLASSES)

	def forward(self, x, edge_index):
		h = self.conv1(x, edge_index)
		h = h.tanh()
		h = self.conv2(h, edge_index)
		h = h.tanh()
		h = self.conv3(h, edge_index)
		h = h.tanh()

		out = self.classifier(h)
		return out, h

model = GCN()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(data):
	optimizer.zero_grad()
	out, h = model(data.x.float(), data.edge_index)
	loss = criterion(out, data.y)
	loss.backward()
	optimizer.step()
	return loss, h

test_size = 10000
test_dataset = [random_graph() for i in range(test_size)]
test_loader = DataLoader(test_dataset, batch_size=1)

correct_nodes = 0
total_nodes = 0
for data in test_loader:
    model.eval()
    out, _ = model(data.x, data.edge_index)
    pred = out.argmax(1)
    test_correct = pred == data.y
    correct_nodes += test_correct.sum()
    total_nodes += data.num_nodes

print(f'Accuracy: {correct_nodes/total_nodes:.3f}')

