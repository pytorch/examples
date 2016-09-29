import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.trainer as trainer
import torch.utils.trainer.plugins
import torch.utils.data

import resnet
import data

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
                    help='mini-batch size (1 = pure stochastic)')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weightDecay', default=1e-4, type=float, metavar='W',
                    help='weight decay')
args = parser.parse_args()

learningRate = args.lr
momentum = args.momentum
cudnn.benchmark = True

if args.arch.startswith('resnet'):
    model = resnet.__dict__[args.arch]()
    model.cuda()
else:
    parser.error('invalid architecture: {}'.format(args.arch))

if torch.cuda.device_count() > 1:
    class DataParallel(nn.Container):
        def __init__(self):
            super(DataParallel, self).__init__(
                model=model,
            )

        def forward(self, input):
            gpu_ids = range(torch.cuda.device_count())
            return nn.parallel.data_parallel(self.model, input, gpu_ids)

    model = DataParallel()

train, val = data.make_datasets(args.data)
train_loader = torch.utils.data.DataLoader(
    train, batch_size=args.batchSize, shuffle=True, num_workers=args.nThreads)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model, learningRate, momentum)
t = trainer.Trainer(model, criterion, optimizer, train_loader)

t.register_plugin(trainer.plugins.AccuracyMonitor())
t.register_plugin(trainer.plugins.LossMonitor())
t.register_plugin(trainer.plugins.TimeMonitor())
t.register_plugin(trainer.plugins.Logger(['accuracy', 'loss', 'time']))
t.run(args.nEpochs)
