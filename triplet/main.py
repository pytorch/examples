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
import random
import numpy as np
import collections

from phototour import PhotoTour

# Training settings
parser = argparse.ArgumentParser(description='PyTorch TFeat Example')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32,
                    help='the height / width of the input image to network')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--n_triplets', type=int, default=1280, metavar='N',
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


class TripletPhotoTour(PhotoTour):
    """From the MNIST Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)

        print('Generating triplets ...')
        self.n_triplets = args.n_triplets
        self.triplets = self.generate_triplets(self.labels)

    '''def generate_triplets(self, labels):
        triplets = []
        labels = labels.numpy()

        ulabels = np.unique(labels)
        matches, no_matches = dict(), dict()
        for x in ulabels:
            matches[x] = np.where(labels == x)[0]
            no_matches[x] = np.where(labels != x)[0]

        candidates = np.random.randint(0, len(labels), size=self.n_triplets)
        candidates = labels[candidates]
        for x in candidates:
            idx_a, idx_p = np.random.choice(matches[x], 2, replace=False)
            idx_n = np.random.choice(no_matches[x], 1)[0]
            triplets.append([idx_a, idx_p, idx_n])
        return np.array(triplets)'''

    def generate_triplets(self, labels):
        def create_indices(_labels):
            """Generates a dict to store the index of each labels in order
               to avoid a linear search each time that we call list(labels).index(x)
            """
            old = labels[0]
            indices = dict()
            indices[old] = 0
            for x in range(len(_labels) - 1):
                new = labels[x + 1]
                if old != new:
                    indices[new] = x + 1
                old = new
            return indices
        triplets = []
        labels = labels.numpy()

        # group labels in order to have O(1) search
        count = collections.Counter(labels)
        # index the labels in order to have O(1) search
        indices = create_indices(labels)
        # range for the sampling
        labels_size = len(labels) - 1
        # generate the triplets
        for x in range(self.n_triplets):
            # pick a random id for anchor
            idx = random.randint(0, len(labels) - 1)
            # count number of anchor occurrences
            num_samples = count[labels[idx]]
            # the global index to the id
            begin_positives = indices[labels[idx]]
            # generate two samples to the id
            offset_a, offset_p = random.sample(range(num_samples), 2)
            idx_a = begin_positives + offset_a
            idx_p = begin_positives + offset_p
            # find index of the same 3D but not same as before
            idx_n = random.randint(0, labels_size)
            while labels[idx_n] == labels[idx_a] and \
                  labels[idx_n] == labels[idx_p]:
                idx_n = random.randint(0, labels_size)
            # pick and append triplets to the buffer
            triplets.append([idx_a, idx_p, idx_n])
        return np.array(triplets)

    def __getitem__(self, index):
        def convert_and_transform(img, transform):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """
            img = Image.fromarray(img.numpy(), mode='L')

            if transform is not None:
                img = self.transform(img)
            return img

        t = self.triplets[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        # transform image if required
        img_a = convert_and_transform(a, self.transform)
        img_p = convert_and_transform(p, self.transform)
        img_n = convert_and_transform(n, self.transform)

        return img_a, img_p, img_n

    def __len__(self):
        return self.triplets.shape[0]


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
    TripletPhotoTour(args.dataroot, name='notredame', download=True,
                     transform=transforms.Compose([
                         transforms.Scale(args.imageSize),
                         transforms.ToTensor(),
                         transforms.Normalize((0.4854,), (0.1864,))
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
