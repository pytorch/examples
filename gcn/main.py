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
    def __init__(self, input_dim, output_dim, use_bias=False):
        super(GraphConv, self).__init__()

        self.kernel = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_normal_(self.kernel)

        self.bias = None
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.bias)

    def forward(self, input_tensor, adj_mat):
        support = torch.mm(input_tensor, self.kernel)
        output = torch.spmm(adj_mat, support)
        if self.bias is not None:
            output = output + self.bias

        return output


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_bias=True, dropout_p=0.1):
        super(GCN, self).__init__()
        self.gc1 = GraphConv(input_dim, hidden_dim, use_bias=use_bias)
        self.gc2 = GraphConv(hidden_dim, output_dim, use_bias=use_bias)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_tensor, adj_mat):
        x = self.gc1(input_tensor, adj_mat)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gc2(x, adj_mat)

        return F.log_softmax(x, dim=1)


def load_cora(path='./cora', device='cpu'):
    content_path = os.path.join(path, 'cora.content')
    cites_path = os.path.join(path, 'cora.cites')

    content_tensor = np.genfromtxt(content_path, dtype=np.dtype(str))
    cites_tensor = np.genfromtxt(cites_path, dtype=np.int32)

    features = torch.FloatTensor(content_tensor[:, 1:-1].astype(np.int32))
    scale_vector = torch.sum(features, dim=1)
    scale_vector = 1 / scale_vector
    scale_vector[scale_vector == float('inf')] = 0
    scale_vector = torch.diag(scale_vector).to_sparse()
    features = scale_vector @ features

    classes, labels = np.unique(content_tensor[:, -1], return_inverse=True)
    labels = torch.LongTensor(labels)

    idx = content_tensor[:, 0].astype(np.int32)
    idx_map = {id: pos for pos, id in enumerate(idx)}

    edges = np.array(
        list(map(lambda edge: [idx_map[edge[0]], idx_map[edge[1]]], 
            cites_tensor)), dtype=np.int32)

    V = len(idx)
    E = edges.shape[0]
    adj_mat = torch.sparse_coo_tensor(edges.T, torch.ones(E), (V, V), dtype=torch.int64)
    adj_mat = torch.eye(V) + adj_mat

    degree_mat = torch.sum(adj_mat, dim=1)
    degree_mat = torch.sqrt(1 / degree_mat)
    degree_mat[degree_mat == float('inf')] = 0
    degree_mat = torch.diag(degree_mat).to_sparse()

    adj_mat = degree_mat @ adj_mat @ degree_mat

    return features.to_sparse().to(device), labels.to(device), adj_mat.to_sparse().to(device)


def train_iter(epoch, model, optimizer, criterion, input, target, mask_train, mask_val, print_every=10):
    start_t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(*input)
    loss = criterion(output[mask_train], target[mask_train])

    loss.backward()
    optimizer.step()

    loss_train, acc_train = test(model, criterion, input, target, mask_train)
    loss_val, acc_val = test(model, criterion, input, target, mask_val)

    if epoch % print_every == 0:
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
    parser.add_argument('--include-bias', type=bool, default=False,
                        help='use bias term in convolutions (default: False)')
    parser.add_argument('--no-cuda', type=bool, default=False,
                        help='disable CUDA training (default: False)')
    args = parser.parse_args()

    device = 'cpu' if args.no_cuda else device
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

    gcn = GCN(features.shape[1], args.hidden_dim, labels.max().item() + 1,args.include_bias, args.dropout_p).to(device)
    optimizer = Adam(gcn.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.NLLLoss()

    for epoch in range(args.epochs):
        train_iter(epoch + 1, gcn, optimizer, criterion, (features, adj_mat), labels, idx_train, idx_val, args.val_every)
    
    loss_test, acc_test = test(gcn, criterion, (features, adj_mat), labels, idx_test)
    print(f'Test set results: loss {loss_test:.4f} accuracy {acc_test:.4f}')