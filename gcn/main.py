import os
import time
import requests
import tarfile
import numpy as np
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam


class GraphConv(nn.Module):
    """
        Graph Convolutional Layer described in "Semi-Supervised Classification with Graph Convolutional Networks".

        Given an input feature representation for each node in a graph, the Graph Convolutional Layer aims to aggregate
        information from the node's neighborhood to update its own representation. This is achieved by applying a graph
        convolutional operation that combines the features of a node with the features of its neighboring nodes.

        Mathematically, the Graph Convolutional Layer can be described as follows:

            H' = f(D^(-1/2) * A * D^(-1/2) * H * W)

        where:
            H: Input feature matrix with shape (N, F_in), where N is the number of nodes and F_in is the number of 
                input features per node.
            A: Adjacency matrix of the graph with shape (N, N), representing the relationships between nodes.
            W: Learnable weight matrix with shape (F_in, F_out), where F_out is the number of output features per node.
            D: The degree matrix.
    """
    def __init__(self, input_dim, output_dim, use_bias=False):
        super(GraphConv, self).__init__()

        # Initialize the weight matrix W (in this case called `kernel`)
        self.kernel = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_normal_(self.kernel) # Initialize the weights using Xavier initialization

        # Initialize the bias (if use_bias is True)
        self.bias = None
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.bias) # Initialize the bias to zeros

    def forward(self, input_tensor, adj_mat):
        """
        Performs a graph convolution operation.

        Args:
            input_tensor (torch.Tensor): Input tensor representing node features.
            adj_mat (torch.Tensor): Normalized adjacency matrix representing graph structure.

        Returns:
            torch.Tensor: Output tensor after the graph convolution operation.
        """

        support = torch.mm(input_tensor, self.kernel) # Matrix multiplication between input and weight matrix
        output = torch.spmm(adj_mat, support) # Sparse matrix multiplication between adjacency matrix and support
        # Add the bias (if bias is not None)
        if self.bias is not None:
            output = output + self.bias

        return output


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) as described in the paper `"Semi-Supervised Classification with Graph 
    Convolutional Networks" <https://arxiv.org/pdf/1609.02907.pdf>`.

    The Graph Convolutional Network is a deep learning architecture designed for semi-supervised node 
    classification tasks on graph-structured data. It leverages the graph structure to learn node representations 
    by propagating information through the graph using graph convolutional layers.

    The original implementation consists of two stacked graph convolutional layers. The ReLU activation function is 
    applied to the hidden representations, and the Softmax activation function is applied to the output representations.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, use_bias=True, dropout_p=0.1):
        super(GCN, self).__init__()

        # Define the Graph Convolution layers
        self.gc1 = GraphConv(input_dim, hidden_dim, use_bias=use_bias)
        self.gc2 = GraphConv(hidden_dim, output_dim, use_bias=use_bias)

        # Define the dropout layer
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_tensor, adj_mat):
        """
        Performs forward pass of the Graph Convolutional Network (GCN).

        Args:
            input_tensor (torch.Tensor): Input node feature matrix with shape (N, input_dim), where N is the number of nodes
                and input_dim is the number of input features per node.
            adj_mat (torch.Tensor): Normalized adjacency matrix of the graph with shape (N, N), representing the relationships between
                nodes.

        Returns:
            torch.Tensor: Output tensor with shape (N, output_dim), representing the predicted class probabilities for each node.
        """

        # Perform the first graph convolutional layer
        x = self.gc1(input_tensor, adj_mat)
        x = F.relu(x) # Apply ReLU activation function
        x = self.dropout(x) # Apply dropout regularization

        # Perform the second graph convolutional layer
        x = self.gc2(x, adj_mat)

        # Apply log-softmax activation function for classification
        return F.log_softmax(x, dim=1)


def load_cora(path='./cora', device='cpu'):
    """
    The graph convolutional operation rquires the normalized adjacency matrix: D^(-1/2) * A * D^(-1/2). This step 
    scales the adjacency matrix such that the features of neighboring nodes are weighted appropriately during 
    aggregation. The steps involved in the renormalization trick are as follows:
        - Compute the degree matrix.
        - Compute the inverse square root of the degree matrix.
        - Multiply the inverse square root of the degree matrix with the adjacency matrix.
    """

    # Set the paths to the data files
    content_path = os.path.join(path, 'cora.content')
    cites_path = os.path.join(path, 'cora.cites')

    # Load data from files
    content_tensor = np.genfromtxt(content_path, dtype=np.dtype(str))
    cites_tensor = np.genfromtxt(cites_path, dtype=np.int32)

    # Process features
    features = torch.FloatTensor(content_tensor[:, 1:-1].astype(np.int32)) # Extract feature values
    scale_vector = torch.sum(features, dim=1) # Compute sum of features for each node
    scale_vector = 1 / scale_vector # Compute reciprocal of the sums
    scale_vector[scale_vector == float('inf')] = 0 # Handle division by zero cases
    scale_vector = torch.diag(scale_vector).to_sparse() # Convert the scale vector to a sparse diagonal matrix
    features = scale_vector @ features # Scale the features using the scale vector

    # Process labels
    classes, labels = np.unique(content_tensor[:, -1], return_inverse=True) # Extract unique classes and map labels to indices
    labels = torch.LongTensor(labels) # Convert labels to a tensor

    # Process adjacency matrix
    idx = content_tensor[:, 0].astype(np.int32) # Extract node indices
    idx_map = {id: pos for pos, id in enumerate(idx)} # Create a dictionary to map indices to positions

    # Map node indices to positions in the adjacency matrix
    edges = np.array(
        list(map(lambda edge: [idx_map[edge[0]], idx_map[edge[1]]], 
            cites_tensor)), dtype=np.int32)

    V = len(idx) # Number of nodes
    E = edges.shape[0] # Number of edges
    adj_mat = torch.sparse_coo_tensor(edges.T, torch.ones(E), (V, V), dtype=torch.int64) # Create the initial adjacency matrix as a sparse tensor
    adj_mat = torch.eye(V) + adj_mat # Add self-loops to the adjacency matrix

    degree_mat = torch.sum(adj_mat, dim=1) # Compute the sum of each row in the adjacency matrix (degree matrix)
    degree_mat = torch.sqrt(1 / degree_mat) # Compute the reciprocal square root of the degrees
    degree_mat[degree_mat == float('inf')] = 0 # Handle division by zero cases
    degree_mat = torch.diag(degree_mat).to_sparse() # Convert the degree matrix to a sparse diagonal matrix

    adj_mat = degree_mat @ adj_mat @ degree_mat # Apply the renormalization trick

    return features.to_sparse().to(device), labels.to(device), adj_mat.to_sparse().to(device)


def train_iter(epoch, model, optimizer, criterion, input, target, mask_train, mask_val, print_every=10):
    start_t = time.time()
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(*input) 
    loss = criterion(output[mask_train], target[mask_train]) # Compute the loss using the training mask

    loss.backward()
    optimizer.step()

    # Evaluate the model performance on training and validation sets
    loss_train, acc_train = test(model, criterion, input, target, mask_train)
    loss_val, acc_val = test(model, criterion, input, target, mask_val)

    if epoch % print_every == 0:
        # Print the training progress at specified intervals
        print(f'Epoch: {epoch:04d} ({(time.time() - start_t):.4f}s) loss_train: {loss_train:.4f} acc_train: {acc_train:.4f} loss_val: {loss_val:.4f} acc_val: {acc_val:.4f}')


def test(model, criterion, input, target, mask):
    model.eval()
    with torch.no_grad():
        output = model(*input)
        output, target = output[mask], target[mask]

        loss = criterion(output, target)
        acc = (output.argmax(dim=1) == target).float().sum() / len(target)
    return loss.item(), acc.item()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(description='PyTorch Graph Convolutional Network')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--l2', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--dropout-p', type=float, default=0.5,
                        help='dropout probability (default: 0.5)')
    parser.add_argument('--hidden-dim', type=int, default=16,
                        help='dimension of the hidden representation (default: 16)')
    parser.add_argument('--val-every', type=int, default=20,
                        help='epochs to wait for print training and validation evaluation (default: 20)')
    parser.add_argument('--include-bias', action='store_true', default=False,
                        help='use bias term in convolutions (default: False)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device('cuda')
    elif use_mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using {device} device')

    cora_url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
    print('Downloading dataset...')
    with requests.get(cora_url, stream=True) as tgz_file:
        with tarfile.open(fileobj=tgz_file.raw, mode='r:gz') as tgz_object:
            tgz_object.extractall()

    print('Loading dataset...')
    features, labels, adj_mat = load_cora(device=device)
    idx = torch.randperm(len(labels)).to(device)
    idx_test, idx_val, idx_train = idx[:1000], idx[1000:1500], idx[1500:]

    gcn = GCN(features.shape[1], args.hidden_dim, labels.max().item() + 1, args.include_bias, args.dropout_p).to(device)
    optimizer = Adam(gcn.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.NLLLoss()

    for epoch in range(args.epochs):
        train_iter(epoch + 1, gcn, optimizer, criterion, (features, adj_mat), labels, idx_train, idx_val, args.val_every)
        if args.dry_run:
            break

    loss_test, acc_test = test(gcn, criterion, (features, adj_mat), labels, idx_test)
    print(f'Test set results: loss {loss_test:.4f} accuracy {acc_test:.4f}')