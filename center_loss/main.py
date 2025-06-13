from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from face_model import FaceModel
from data import ImageDataset
import trainer as tn
import math
import os
import copy
# Training settings
parser = argparse.ArgumentParser(description='PyTorch face recognition Example')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test_batch_size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--max_iter', type=int, default=100000, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--center_loss_weight', type=float, default=0.1, help='weight for center loss')
parser.add_argument('--root', type=str,
        help='path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='./datasets/casia_mtcnnpy_160')
parser.add_argument('--resume', type=str,
        help='model path to the resume training',
        default='')

# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

def main():
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    # 1. dataset
    root = args.root
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_transforms = transforms.Compose([transforms.Scale((96,128)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),])
    test_transforms = transforms.Compose([transforms.Scale((96,128)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
    train_dataset = ImageDataset(root, transform=train_transforms)
    val_dataset = copy.deepcopy(train_dataset)
    val_dataset.train = False

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # 2. model
    model = FaceModel(len(train_dataset.classes))
    if cuda:
        model = model.cuda()

    # 3. optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    trainer = tn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        max_iter=args.max_iter,
        center_loss_weight = args.center_loss_weight
    )

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            trainer.epoch = checkpoint['epoch']
            trainer.iteration =  trainer.epoch * len(train_loader)
            trainer.best_prec1 = checkpoint['best_prec1']
            trainer.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model.apply(weights_init)
        trainer.epoch = 0
        trainer.iteration = 0
        trainer.best_prec1 = 0
    try:
        trainer.train()
    except:
        raise

if __name__ == '__main__':
    main()
