import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.trainer as trainer
import torch.utils.trainer.plugins
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import resnet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='PATH', required=True,
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture: resnet18 | resnet34 | ...'
                         '(default: resnet18)')
parser.add_argument('--gen', default='gen', metavar='PATH',
                    help='path to save generated files (default: gen)')
parser.add_argument('--nThreads', '-j', default=2, type=int, metavar='N',
                    help='number of data loading threads (default: 2)')
parser.add_argument('--nEpochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epochNumber', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batchSize', '-b', default=256, type=int, metavar='N',
                    help='mini-batch size (1 = pure stochastic) Default: 256')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weightDecay', default=1e-4, type=float, metavar='W',
                    help='weight decay')
args = parser.parse_args()

if args.arch.startswith('resnet'):
    model = resnet.__dict__[args.arch]()
    model.cuda()
else:
    parser.error('invalid architecture: {}'.format(args.arch))

cudnn.benchmark = True

# Data loading code
transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ]),
])

traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')
train = datasets.ImageFolder(traindir, transform)
val = datasets.ImageFolder(valdir, transform)
train_loader = torch.utils.data.DataLoader(
    train, batch_size=args.batchSize, shuffle=True, num_workers=args.nThreads)


# create a small container to apply DataParallel to the ResNet
class DataParallel(nn.Container):
    def __init__(self):
        super(DataParallel, self).__init__(
            model=model,
        )

    def forward(self, input):
        if torch.cuda.device_count() > 1:
            gpu_ids = range(torch.cuda.device_count())
            return nn.parallel.data_parallel(self.model, input, gpu_ids)
        else:
            return self.model(input.cuda()).cpu()

model = DataParallel()

# define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), args.lr, args.momentum)


# pass model, loss, optimizer and dataset to the trainer
t = trainer.Trainer(model, criterion, optimizer, train_loader)

# register some monitoring plugins
t.register_plugin(trainer.plugins.ProgressMonitor())
t.register_plugin(trainer.plugins.AccuracyMonitor())
t.register_plugin(trainer.plugins.LossMonitor())
t.register_plugin(trainer.plugins.TimeMonitor())
t.register_plugin(trainer.plugins.Logger(['progress', 'accuracy', 'loss', 'time']))

# train!
t.run(args.nEpochs)
