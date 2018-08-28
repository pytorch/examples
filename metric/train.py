# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
import os
import sys
import time
from torch.backends import cudnn
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from losses import TripletLoss
from utils import RandomIdentitySampler, mkdir_if_missing
from evaluation import nmi, recall 
import DataSet

cudnn.benchmark = True

model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-data', default='car', required=True,
                    help='path to dataset')
parser.add_argument('-root', default='', type=str,
                    help='image root dir')
parser.add_argument('-margin', default=0.5, type=float, required=False,
                   help='margin in loss function')
parser.add_argument('-arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-start', default=0, type=int,
                    help='resume epoch')
parser.add_argument('-print_freq', default=10, type=int,
                    help='print freq ')
parser.add_argument('-save_dir', default=None,
                    help='where the trained models save')
parser.add_argument('-batch_size', '-b', default=128, type=int, metavar='N',
                    help='mini-batch size (1 = pure stochastic) Default: 256')
parser.add_argument('-num_instances', default=4, type=int, metavar='n',
                    help='the number of samples from one class in mini-batch')
parser.add_argument('-epochs', default=400, type=int, metavar='N',
                    help='epochs for training process')
parser.add_argument('-lr_step', default=100, type=int, metavar='N',
                    help='number of epochs to decay learn rate')
parser.add_argument('-test_step', default=100, type=int, metavar='N',
                    help='number of epochs to test')
parser.add_argument('-save_step', default=40, type=int, metavar='N',
                    help='number of epochs to save model')
# optimizer
parser.add_argument('-lr', type=float, default=1e-4,
                    help="learning rate of new parameters")
parser.add_argument('-nThreads', '-j', default=4, type=int, metavar='N',
                    help='number of data loading threads (default: 2)')
parser.add_argument('-momentum', type=float, default=0.9)
parser.add_argument('-weight-decay', type=float, default=2e-4)
parser.add_argument('-orth_cof', type=float, default=0e-3,
                    help='try to make the last linear weight matrix to '
                         'approximate the orthogonal matrix')

def main():
    global args
    args = parser.parse_args()

    if args.save_dir is None:
        save_dir = os.path.join('checkpoints', args.loss)
    else:
        save_dir = os.path.join('checkpoints', args.save_dir)
    mkdir_if_missing(save_dir)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, num_classes=1000) #for resnet
        #model.fc = nn.Linear(model.fc.in_features, 1000)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    model = models.Net(model) 
    model.cuda() # one machine multi-gpus with data parallel
    model = torch.nn.DataParallel(model) # one machine multi-gpus with data parallel

    criterion = TripletLoss(margin=args.margin).cuda() 

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir, 
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=RandomIdentitySampler(train_dataset, num_instances=args.num_instances),
        pin_memory=True, drop_last=True, num_workers=args.nThreads)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        sampler=RandomIdentitySampler(val_dataset, num_instances=args.num_instances),
        pin_memory=True, drop_last=True, num_workers=args.nThreads)

    for epoch in range(args.start, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch)
        
        if epoch % args.test_step == 0: 
            validate(val_loader, model, criterion, epoch)

        if epoch % args.save_step == 0:
            torch.save(model, os.path.join(save_dir, '%d_model.pkl' % epoch))

    torch.save(model, os.path.join(save_dir, '%d_model.pkl' % epoch))
    print('Finished Training')


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    triplet_loss = AverageMeter()
    
    for i, (input, target) in enumerate(train_loader):
        start = time.time() 
        target = target.cuda(async=True)
        input = Variable(input)
        target = Variable(target)
       
        # forward + backward + optimize
        embed_feat = model(input)
        loss = criterion(embed_feat, target)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        triplet_loss.update(loss.data[0], input.size(0))
        batch_time = time.time() - start

        if i % args.print_freq == 0:
            cur = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            lr = optimizer.param_groups[0]['lr']
            print('%s -> Train Epoch: [%d][%d/%d]\t'
                  'lr: %.8f\tbatch_time: %.3f\t'
                  'loss.val: %.7f\tloss.avg: %.7f\t' % (
                   cur, epoch, i, len(train_loader), lr, batch_time, 
                   triplet_loss.val, triplet_loss.avg), flush=True)
        

def validate(val_loader, model, criterion, epoch):
    model.train()
    triplet_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
 
    for i, (input, target) in enumerate(val_loader):
        start = time.time() 
        target = target.cuda(async=True)
        input = Variable(input)
        target = Variable(target)
       
        # forward 
        embed_feat = model(input)
        loss = criterion(embed_feat, target)
        
        # nmi + recall
        embed_feat = embed_feat.data.cpu()
        target = target.data.cpu()
        num_classes = len(set(target))
        nmi_score = nmi(embed_feat, target, num_classes)
        prec1, prec5 = recall(embed_feat, target)
        
        triplet_loss.update(loss.data[0], input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        batch_time = time.time() - start

        cur = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        print('%s -> Test Epoch: [%d][%d/%d]\t'
              'batch_time: %.3f\t'
              'loss.val: %.7f\tloss.avg: %.7f\t' % (
               cur, epoch, i, len(val_loader), batch_time, 
               triplet_loss.val, triplet_loss.avg), flush=True)
        print('\t\tNMI score: %.2f\tRecall@1: %.2f\tRecall@5: %.2f' % (nmi_score, prec1, prec5))
    
    print('\t * Test -> NMI score: %.2f\tRecall@1: %.2f\tRecall@5: %.2f' % (nmi_score, top1.avg, top5.avg))
        

def test(data_loader, model):
    print('Test:')
    features, labels = extract_features(model, data_loader, print_freq=100)
    num_class = len(set(labels))
    print('\tNMI score:', NMI(features, labels, n_cluster=num_class))

    sim_mat = - pairwise_distance(features)
    recall_at_ks = Recall_at_ks(sim_mat, query_ids=labels, gallery_ids=labels)
    print('\tRecall@K:', recall_at_ks)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // args.lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
