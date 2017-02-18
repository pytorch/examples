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
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.autograd import Function
import torch.backends.cudnn as cudnn

import os
import cv2
import numpy as np
from tqdm import tqdm

from phototour import PhotoTour
from eval_metrics import ErrorRateAt95Recall

from tensorboard_logger import configure, log_value

# Training settings
parser = argparse.ArgumentParser(description='PyTorch TFeat Example')
# Model options
parser.add_argument('--dataroot', type=str, default='/tmp/phototour_dataset',
                    help='path to dataset')
parser.add_argument('--log-dir', default='./logs',
                    help='folder to output model checkpoints')
parser.add_argument('--imageSize', type=int, default=32,
                    help='the height / width of the input image to network')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
# Training options
parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='BST',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--n-triplets', type=int, default=1280000, metavar='N',
                    help='how many triplets will generate from the dataset')
parser.add_argument('--margin', type=float, default=2.0, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 2.0')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--lr-decay', default=1e-6, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-6')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: SGD)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
cv2.setRNGSeed(args.seed)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if args.cuda:
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

LOG_DIR = args.log_dir + '/run-optim_{}-n{}-lr{}-wd{}-m{}-S{}-tanh'\
    .format(args.optimizer, args.n_triplets, args.lr, args.wd,
            args.margin, args.seed)


class Logger(object):
    def __init__(self, log_dir):
        # clean previous logged data under the same directory name
        self._remove(log_dir)

        # configure the project
        configure(log_dir)

        self.global_step = 0

    def log_value(self, name, value):
        log_value(name, value, self.global_step)
        return self

    def step(self):
        self.global_step += 1

    @staticmethod
    def _remove(path):
        """ param <path> could either be relative or absolute. """
        if os.path.isfile(path):
            os.remove(path)  # remove the file
        elif os.path.isdir(path):
            import shutil
            shutil.rmtree(path)  # remove dir and all contains

# create logger
logger = Logger(LOG_DIR)


class TripletPhotoTour(PhotoTour):
    """From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, train=True, transform=None, *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)
        self.transform = transform

        self.train = train
        self.n_triplets = args.n_triplets

        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.labels, self.n_triplets)

    @staticmethod
    def generate_triplets(labels, num_triplets):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        triplets = []
        indices = create_indices(labels)
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]

        for x in tqdm(range(num_triplets)):
            c1 = np.random.randint(0, n_classes-1)
            c2 = np.random.randint(0, n_classes-1)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes-1)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]) - 1)
                n2 = np.random.randint(0, len(indices[c1]) - 1)
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]) - 1)
            n3 = np.random.randint(0, len(indices[c2]) - 1)

            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        return torch.LongTensor(np.array(triplets))

    def __getitem__(self, index):
        def transform(img):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """
            if transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            m = self.matches[index]
            img1, img2 = transform(self.data[m[0]]), transform(self.data[m[1]])
            return img1, img2, m[2]

        t = self.triplets[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        # transform images if required
        img_a, img_p, img_n = transform(a), transform(p), transform(n)
        return img_a, img_p, img_n

    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return self.matches.size(0)


class TNet(nn.Module):
    """TFeat model definition
    """
    def __init__(self):
        super(TNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=6),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def weights_init(m):
    # From https://github.com/alykhantejani/nninit/blob/master/nninit.py#L66
    def _calculate_fan_in_and_fan_out(tensor):
        if tensor.ndimension() < 2:
            raise ValueError(
               "fan in and fan out can not be computed for tensor of size ",
               tensor.size())

        if tensor.ndimension() == 2:  # Linear
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = np.prod(tensor.numpy().shape[2:])
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def xavier_uniform(data, gain=1):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(data)
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        a = np.sqrt(3.0) * std
        data.uniform_(-a, a)

    def constant(tensor, val):
        tensor.fill_(val)

    if isinstance(m, nn.Conv2d):
        xavier_uniform(m.weight.data)
        constant(m.bias.data, 0.1)


class PairwiseDistance(Function):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)


class TripletMarginLoss(Function):
    """Triplet loss function.
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)  # norm 2

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, negative)

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss


class TripletMarginLossSwap(Function):
    """Triplet loss function.
    """
    def __init__(self, margin):
        super(TripletMarginLossSwap, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, negative)
        d_n_ = self.pdist.forward(positive, negative)

        d_s = torch.min(d_n, d_n_)

        dist_hinge = torch.clamp(self.margin + d_p - d_s, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss


def triplet_loss(input1, input2, input3, margin=1.0):
    """Interface to call TripletMarginLoss
    """
    return TripletMarginLoss(margin).forward(input1, input2, input3)


def triplet_loss_swap(input1, input2, input3, margin=1.0):
    """Interface to call TripletMarginLoss
    """
    return TripletMarginLossSwap(margin).forward(input1, input2, input3)

cv2_scale = lambda x: cv2.resize(x, dsize=(32, 32),
                                 interpolation=cv2.INTER_LINEAR)
np_reshape = lambda x: np.reshape(x, (1, 32, 32))

kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    TripletPhotoTour(train=True, root=args.dataroot, name='notredame',
                     download=True, size=32,
                     transform=transforms.Compose([
                         transforms.Lambda(cv2_scale),
                         transforms.Lambda(np_reshape),
                         transforms.ToTensor(),
                         transforms.Normalize((0.48544601108437,), (0.18649942105166,))
                     ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    TripletPhotoTour(train=False, root=args.dataroot, name='liberty',
                     download=True, size=32,
                     transform=transforms.Compose([
                         transforms.Lambda(cv2_scale),
                         transforms.Lambda(np_reshape),
                         transforms.ToTensor(),
                         transforms.Normalize((0.48544601108437,), (0.18649942105166,))
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

    optimizer = create_optimizer(model, args.lr)

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

    start = args.start_epoch
    end = start + args.epochs
    for epoch in range(start, end):
        train(train_loader, model, optimizer, epoch)
        test(test_loader, model, epoch)


def train(train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()

    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data_a, data_p, data_n) in pbar:

        if args.cuda:
            data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
        data_a, data_p, data_n = Variable(data_a), Variable(data_p), \
                                 Variable(data_n)

        # compute output
        out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)
        loss = triplet_loss(out_a, out_p, out_n, margin=args.margin)

        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the optimizer learning rate
        adjust_learning_rate(optimizer)

        # log loss value
        logger.log_value('loss', loss.data[0]).step()

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data[0]))

    # do checkpointing
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))


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
    num_tests = test_loader.dataset.matches.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.vstack(distances).reshape(num_tests)

    fpr95 = ErrorRateAt95Recall(labels, distances)
    print('\33[91mTest set: Accuracy(FPR95): {:.8f}\n\33[0m'.format(fpr95))

    logger.log_value('fpr95', fpr95)


def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0
        group['step'] += 1

        group['lr'] = args.lr / (1 + group['step'] * args.lr_decay)


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    return optimizer

if __name__ == '__main__':
    main()