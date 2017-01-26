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
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image
import os
import random
import numpy as np
import collections
from tqdm import tqdm

from phototour import PhotoTour
from eval_metrics import ErrorRateAt95Recall

# Training settings
parser = argparse.ArgumentParser(description='PyTorch TFeat Example')
# Model options
parser.add_argument('--dataroot', type=str, default='/tmp/phototour_dataset',
                    help='path to dataset')
parser.add_argument('--out_dir', default='./logs',
                    help='folder to output model checkpoints')
parser.add_argument('--imageSize', type=int, default=32,
                    help='the height / width of the input image to network')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=60, metavar='E',
                    help='number of epochs to train (default: 10)')
# Training options
parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='BST',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--n_triplets', type=int, default=1280000, metavar='N',
                    help='how many triplets will generate from the dataset')
parser.add_argument('--margin', type=float, default=1.0, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--lrd', default=1e-6, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-6')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu_id', default=1, type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=666, metavar='S',
                    help='random seed (default: 666)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu_id)


class TripletPhotoTour(PhotoTour):
    """From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, train=True, *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)

        self.train = train
        self.n_triplets = args.n_triplets

        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.labels)
            print('Generating {} triplets - done'.format(self.n_triplets))

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

        if not self.train:
            m = self.matches[index]
            img1 = convert_and_transform(self.data[m[0]], self.transform)
            img2 = convert_and_transform(self.data[m[1]], self.transform)
            return img1, img2, m[2]

        t = self.triplets[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        # transform image if required
        img_a = convert_and_transform(a, self.transform)
        img_p = convert_and_transform(p, self.transform)
        img_n = convert_and_transform(n, self.transform)

        return img_a, img_p, img_n

    def __len__(self):
        if self.train:
            return self.triplets.shape[0]
        else:
            return self.matches.shape[0]


class TNet(nn.Module):
    """TFeat model definition
    """
    def __init__(self):
        super(TNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=6)
        self.fc1 = nn.Linear(64*8*8, 128)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.tanh(self.conv2(x))
        x = x.view(-1, 64*8*8)
        x = F.tanh(self.fc1(x))
        return x


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        ni = m.in_channels
        no = m.out_channels
        k, k = m.kernel_size
        std = 2.0/np.sqrt(ni*k*k*no)
        #m.weight.data.normal_(0.0, std)
        m.weight.data.uniform_(-std, std)
        m.bias.data.uniform_(-std, std)


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
    return TripletMarginLoss(margin).forward(input1, input2, input3)


kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    TripletPhotoTour(train=True, root=args.dataroot, name='notredame',
                     download=True,
                     transform=transforms.Compose([
                         transforms.Scale(args.imageSize),
                         transforms.ToTensor(),
                         transforms.Normalize((0.4854,), (0.1864,))
                     ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    TripletPhotoTour(train=False, root=args.dataroot, name='liberty',
                     download=True,
                     transform=transforms.Compose([
                         transforms.Scale(args.imageSize),
                         transforms.ToTensor(),
                         transforms.Normalize((0.4854,), (0.1864,))
                     ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


def main():

    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))

    # instantiate model and initialize weights
    model = TNet()
    model.apply(weights_init)
    if args.cuda:
        model.cuda()

    # setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          dampening=args.momentum,
                          weight_decay=args.weight_decay)
    '''optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)'''

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, optimizer, epoch)
        test(test_loader, model, epoch)


def train(train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()

    loss_train = 0
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data_a, data_p, data_n) in pbar:
        if args.cuda:
            data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
        data_a, data_p, data_n = Variable(data_a), Variable(data_p), \
                                 Variable(data_n)

        # compute output
        out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)
        loss = triplet_loss(out_a, out_p, out_n, margin=args.margin)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        loss_train += loss.data[0]

        # updates the learning rate acording to the original Lua implementation
        adjust_learning_rate(optimizer)
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))

    # reset the momentum buffer
    reset_momentum(optimizer)

    # do checkpointing
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}/checkpoint_{}.pth'.format(args.out_dir, epoch))

    # log the loss to check the learning curve
    write('loss.txt', loss_train / args.batch_size)


def test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a, volatile=True), \
                                Variable(data_p, volatile=True), Variable(label)

        # compute output
        out_a, out_p = model(data_a), model(data_p)
        dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

    # measure accuracy (FPR95)
    num_tests = test_loader.dataset.matches.shape[0]
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.vstack(distances).reshape(num_tests)
    fpr95 = ErrorRateAt95Recall(labels, distances)
    print('Test set: Accuracy(FPR95): {:.4f}\n'.format(fpr95))

    # log the fpr95
    write('fpr95.txt', fpr95)


def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for param_group in optimizer.state_dict()['param_groups']:
        if 'evalCounter' not in param_group:
            param_group['evalCounter'] = 0

        # https://github.com/torch/optim/blob/master/sgd.lua#L72
        param_group['lr'] = args.lr / (1 + param_group['evalCounter'] * args.lrd)

        param_group['evalCounter'] += 1


def reset_momentum(optimizer):
    """Resets to zero the momentum buffer
    note: Extracted from
    https://github.com/szagoruyko/attention-transfer/blob/master/cifar.py#L312-L315
    """
    for param_group in optimizer.state_dict()['param_groups']:
        for p in param_group['params']:
            param_state = optimizer.state[id(p)]
            param_state['momentum_buffer'].zero_()


def write(file_name, value):
    file_path = '{}/{}'.format(args.out_dir, file_name)
    with open(file_path, 'a') as f:
        f.write('{} '.format(value))

if __name__ == '__main__':
    main()