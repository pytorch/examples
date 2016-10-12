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
                    help='mini-batch size (1 = pure stochastic) Default: 256')
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

# this is soon going to go away, and nn.parallel.data_parallel
# will be pushed into the model definition
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

# For debugging. Left in code to leave as an example for the user
# for m in model.modules():
#     def func(self, gi, go):
#         print(go[0].min(), go[0].max())
#         if isinstance(m, nn.Conv2d):
#             print(self.weight.grad.min(), self.weight.grad.max())
#             print(self.bias.grad.min(), self.bias.grad.max())
#     m.register_backward_hook('print_grads', func)

train, val = data.make_datasets(args.data)
train_loader = torch.utils.data.DataLoader(
    train, batch_size=args.batchSize, shuffle=True, num_workers=args.nThreads)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), learningRate, momentum)
t = trainer.Trainer(model, criterion, optimizer, train_loader)

t.register_plugin(trainer.plugins.ProgressMonitor())
t.register_plugin(trainer.plugins.AccuracyMonitor())
t.register_plugin(trainer.plugins.LossMonitor())
t.register_plugin(trainer.plugins.TimeMonitor())
t.register_plugin(trainer.plugins.Logger(['progress', 'accuracy', 'loss', 'time']))
t.run(args.nEpochs)
