import itertools

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.utils import remove_self_loops
import networkx as nx


# NUM_NODES = 6
NUM_FEATURES = 1
NUM_CLASSES = 2
NUM_EPOCHS = 100
BATCH_SIZE = 4
DATASET_SIZE = 256 * BATCH_SIZE
LEARNING_RATE = 0.002


class GCN(torch.nn.Module):

    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = SAGEConv(NUM_FEATURES, 4)
        self.conv2 = SAGEConv(4, 4)
        self.conv3 = SAGEConv(4, 2)
        self.classifier = Linear(2, NUM_CLASSES)

    def forward(self, x, edge_index):
        #print(x)
        #print(edge_index)
        h = self.conv1(x, edge_index)
        #print(h)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = self.conv3(h, edge_index)
        h = F.relu(h)  # Final GNN embedding space.

        # Apply a final (linear) classifier.
        out = self.classifier(h)
        # out = torch.sigmoid(h)

        return out, h


def random_graph(num_nodes=None):
    """ Create a random graph. """

    # Create data.
    if num_nodes is None:
        num_nodes = np.random.randint(5, 10)
    nodes = np.random.rand(num_nodes, 1)
    edge_index = np.reshape(np.meshgrid(np.arange(num_nodes), np.arange(num_nodes)), (2, -1))
    min_index = nodes.argmin(0)

    # Convert to torch data.
    x = torch.tensor(nodes, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)
    y = torch.zeros(num_nodes, dtype=torch.long)
    y[min_index] = 1

    # Create graph object.
    data = Data(x=x, edge_index=edge_index, y=y)

    return data


def visualize(h, color, epoch=None, loss=None):
    """ Visualize a given graph. """

    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)

    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True, node_color=color, cmap="Set2")

    plt.show()


#if __name__ == "__main__":

# Create dataset.
dataset = [random_graph() for i in range(DATASET_SIZE)]
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Instantiate network, loss function, and optimizer.
model = GCN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train network.
for epoch in range(NUM_EPOCHS):
    # Single training batch.
    for data in dataloader:
        optimizer.zero_grad()
        # print(data.x)
        out, h = model(data.x, data.edge_index)
        # print(out, data.y)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        print("Loss: %f" % loss.item())

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
