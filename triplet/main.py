"""
    PyTorch training code for TFeat shallow convolutional patch descriptor:
    http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf

    The code reproduces *exactly* it's lua anf TF version:
    https://github.com/vbalnt/tfeat

    2017 Edgar Riba
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch TFeat Example')
parser.add_argument('--imageSize', type=int, default=32,
                    help='the height / width of the input image to network')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--n_triplets', type=int, default=6400, metavar='N',
                    help='how many triplets will generate from the dataset')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


class TripletMNIST(datasets.MNIST):
    """From the MNIST Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, *arg, **kw):
        super(TripletMNIST, self).__init__(*arg, **kw)

        print('Generating triplets ...')
        self.n_triplets = args.n_triplets
        self.train_triplets = self.generate_triplets(self.train_labels)

    def generate_triplets(self, labels):
        triplets = []
        for x in xrange(self.n_triplets):
            idx = np.random.randint(0, labels.size(0))
            idx_matches = np.where(labels.numpy() == labels[idx])[0]
            idx_no_matches = np.where(labels.numpy() != labels[idx])[0]
            idx_a, idx_p = np.random.choice(idx_matches, 2, replace=False)
            idx_n = np.random.choice(idx_no_matches, 1)[0]
            triplets.append([idx_a, idx_p, idx_n])
        return np.array(triplets)

    def __getitem__(self, index):
        if self.train:
            t = self.train_triplets[index]
            a, p, n = self.train_data[t[0]], self.train_data[t[1]],\
                      self.train_data[t[2]]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img_a = Image.fromarray(a.numpy(), mode='L')
        img_p = Image.fromarray(p.numpy(), mode='L')
        img_n = Image.fromarray(n.numpy(), mode='L')

        if self.transform is not None:
            img_a = self.transform(img_a)
            img_p = self.transform(img_p)
            img_n = self.transform(img_n)

        return img_a, img_p, img_n

    def __len__(self):
        if self.train:
            return self.train_triplets.shape[0]


class TNet(nn.Module):
    """TFeat model definition
    """
    def __init__(self):
        super(TNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64*8*8, 128)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.tanh(self.conv2(x))
        x = x.view(-1, 64*8*8)
        x = F.tanh(self.fc1(x))
        return x


class TripletMarginLoss(nn.Module):
    """Triplet loss function.
    Based on: http://docs.chainer.org/en/stable/_modules/chainer/functions/loss/triplet.html
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        dist = torch.sum(
            (anchor - positive) ** 2 - (anchor - negative) ** 2,
            dim=1) + self.margin
        dist_hinge = torch.clamp(dist, min=0.0)  # maximum between 'dist' and 0.0
        loss = torch.mean(dist_hinge)
        return loss


def triplet_loss(input1, input2, input3, margin=1.0):
    """Interface to call TripletMarginLoss
    """
    return TripletMarginLoss(margin)(input1, input2, input3)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    TripletMNIST('../data', train=True, download=True,
                 transform=transforms.Compose([
                     transforms.Scale(args.imageSize),
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307,), (0.3081,))
                 ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


model = TNet()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    model.train()
    for batch_idx, (data_a, data_p, data_n) in enumerate(train_loader):
        if args.cuda:
            data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
        data_a, data_p, data_n = Variable(data_a), Variable(data_p), Variable(data_n)
        optimizer.zero_grad()
        out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)
        loss = triplet_loss(out_a, out_p, out_n)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_a), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(epoch)