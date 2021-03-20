import itertools
import random

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
import math

flag_training = False

MIN_NUM_NODES = 4
MAX_NUM_NODES = 10
NUM_FEATURES = 1
HIDDEN_SIZE = 16
#NUM_NODE_CLASSES = 2
#NUM_EDGE_CLASSES = 2
NUM_EPOCHS = 1000
BATCH_SIZE = 16
DATASET_SIZE = 64 * BATCH_SIZE
LEARNING_RATE = 0.001
SEED = 1
TRAIN_SPLIT = 0.8


class GCN(torch.nn.Module):

    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.conv1 = SAGEConv(NUM_FEATURES, HIDDEN_SIZE)
        self.conv2 = SAGEConv(HIDDEN_SIZE, HIDDEN_SIZE)
        self.conv3 = SAGEConv(HIDDEN_SIZE, HIDDEN_SIZE)
        self.edge_fc = Linear(2*HIDDEN_SIZE+2, 2*HIDDEN_SIZE+2)

        # Initialize classifiers
        self.node_cls = Linear(HIDDEN_SIZE, 1)
        self.edge_cls = Linear(2*HIDDEN_SIZE+2, 1)

    def forward(self, x, edge_index):
        # Embedding
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = self.conv3(h, edge_index)
        h = F.relu(h)  # Final GNN embedding space.

        # Apply a final (linear) classifier.
        node_out = self.node_cls(h)
        #print('h.shpae = {}'.format(h.shape))
        #print('node_out.shape = {}'.format(node_out.shape))
        src, dst = edge_index

        edge_repr = torch.cat([x[src], x[dst], h[src], h[dst]], dim=-1)
        edge_h = self.edge_fc(edge_repr)
        edge_h = F.relu(edge_h)
        #print('edge_h.shpae = {}'.format(edge_h.shape))
        #print('node_out.shape = {}'.format(node_out))
        edge_out = self.edge_cls(edge_h)

        return node_out, edge_out


def random_graph(num_nodes=None):
    """ Create a random graph. """

    # Create data.
    nodes = np.random.rand(num_nodes, 1)
    #nodes = np.sort(nodes, axis=0)
    sorted_idx = np.argsort(nodes, axis=0)

    edge_index = []
    edge_attr = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            edge_index.append((i, j))
            # edge_attr denote if nodes[j] is the next node in the sorted array
            edge_attr.append(int(sorted_idx[i]+1 == sorted_idx[j]))

    edge_index = np.array(edge_index).T 
    edge_attr  = np.array(edge_attr)
    min_index  = nodes.argmin(0)

    # Convert to torch data.
    x = torch.tensor(nodes)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr  = torch.tensor(edge_attr, dtype=torch.long)
    y = torch.zeros(num_nodes, dtype=torch.long)
    y[min_index] = 1

    # Create graph object
    data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)

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

def visualize_edges(edge_out):
    edge_prob_pairs = F.softmax(edge_out, dim=-1)
    print('edge_prob_pairs.shape = {}'.format(edge_prob_pairs.shape))
    num_nodes = int(math.sqrt(edge_out.shape[0]))
    print('num_nodes = {}'.format(num_nodes))
    edge_probs = edge_prob_pairs[:, 1].reshape(num_nodes, num_nodes)
    plt.imshow(edge_probs.detach().numpy(), vmin=0, vmax=1)
    plt.show()


#if __name__ == "__main__":

# Set seed
np.random.seed(SEED)
torch.manual_seed(SEED) 
torch.cuda.manual_seed_all(SEED)

# Create dataset.
dataset = [
    random_graph(random.choice(range(MIN_NUM_NODES, MAX_NUM_NODES))) 
    for i in range(DATASET_SIZE)
]
train_dataset_size = int(len(dataset) * TRAIN_SPLIT)
train_dataset = dataset[:train_dataset_size]
test_dataset  = dataset[train_dataset_size:]
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader  = DataLoader(test_dataset, batch_size=1)

# Instantiate network, loss function, and optimizer.
model_dir = 'gcn_sorting_epoch_{}.pth'.format(NUM_EPOCHS)
model = GCN().cuda()
if model_dir != None:
    model.load_state_dict(torch.load(model_dir))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)

# Train model
if flag_training:
    print('=====================================Training=====================================')
    for epoch in range(NUM_EPOCHS):
        print('epoch = {}'.format(epoch))
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            assert 1==2
            node_out, edge_out = model(data.x.float().cuda(), data.edge_index.cuda())
            #assert 1==2
            node_loss = criterion(node_out.squeeze(1), data.y.cuda().float())
            edge_loss = criterion(edge_out.squeeze(1), data.edge_attr.cuda().float())
            loss = node_loss + edge_loss
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('iter = {}, Node, edge, total loss: {:.4f}, {:.4f}, {:.4f}'.format(i, node_loss.item(), edge_loss.item(), loss.item()))

        scheduler.step(loss.item())

    torch.save(model.state_dict(), 'gcn_sorting_epoch_{}.pth'.format(NUM_EPOCHS))

#assert 1==2

print('==================================== Testing ============================================')
model.eval()
count_correct = 0
with torch.no_grad():
    for data in test_dataloader:
        '''
        node_out, edge_out = model(data.x.float(), data.edge_index)
        node_loss = criterion(node_out, data.y)
        edge_loss = criterion(edge_out, data.edge_attr)
        loss = node_loss + edge_loss
        visualize_edges(edge_out)
        '''

        out, edge_out = model(data.x.float().cuda(), data.edge_index.cuda())
        out = out.cpu().numpy().squeeze(1)
        edge_out = edge_out.cpu().numpy().squeeze(1)

        nodes = data.x.cpu().numpy()
        gt_sorted_idx = np.argsort(nodes, axis=0).squeeze(1) # ground truth sorted array

        num_nodes = nodes.shape[0]
        pred_sorted_idx_arr = np.ones((num_nodes), dtype=int)
        start_idx = np.argmax(out)
        gt_start_idx = np.argmax(data.y.numpy())
        print('start_idx = {}, gt = {}'.format(start_idx, gt_start_idx))
        #if start_idx == gt_start_idx:
        #    count_correct += 1
        #'''
        pred_sorted_idx_arr[0] = start_idx

        edge_out = edge_out.reshape((num_nodes, num_nodes))
        for i in range(num_nodes-1):
            next_idx = np.argmax(edge_out[i])
            pred_sorted_idx_arr[i+1] = next_idx
        a = (pred_sorted_idx_arr == gt_sorted_idx)
        if np.sum(a) == num_nodes:
            count_correct += 1
        #'''




print(f'Accuracy: {count_correct/len(test_dataloader):.3f}')

'''
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
    pred(1)
    test_correct = pred == data.y
    correct_nodes += test_correct.sum()
    total_nodes += data.num_nodes

print(f'Accuracy: {correct_nodes/total_nodes:.3f}')
'''