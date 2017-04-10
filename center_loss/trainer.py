import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import os
import itertools
import shutil
import tqdm
import numpy as np

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

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

def get_center_loss(centers, features, target, alpha, num_classes):
    batch_size = target.size(0)
    features_dim = features.size(1)

    target_expand = target.view(batch_size,1).expand(batch_size,features_dim)
    centers_var = Variable(centers)
    centers_batch = centers_var.gather(0,target_expand)
    criterion = nn.MSELoss()
    center_loss = criterion(features,  centers_batch)

    diff = centers_batch - features
    unique_label, unique_reverse, unique_count = np.unique(target.cpu().data.numpy(), return_inverse=True, return_counts=True)
    appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))
    appear_times_expand = appear_times.view(-1,1).expand(batch_size,features_dim).type(torch.FloatTensor)
    diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)
    diff_cpu = alpha * diff_cpu
    for i in range(batch_size):
        centers[target.data[i]] -= diff_cpu[i].type(centers.type())

    return center_loss, centers

class Trainer(object):
    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, max_iter):
        self.cuda = cuda
        self.model = model
        self.optim = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.criterion = nn.CrossEntropyLoss()
        self.top1 = AverageMeter()
        self.losses = AverageMeter()
        self.best_prec1 = 0

    def validate(self):
        self.model.eval()
        top1 = AverageMeter()
        losses = AverageMeter()
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Testing on epoch %d' % self.epoch, ncols=80, leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda(async=True)
                self.model.centers = self.model.centers.cuda()

            data_var = Variable(data, volatile=True)
            target_var = Variable(target, volatile=True)
            target_var = Variable(target, volatile=True)
            # compute output
            output = self.model(data_var)
            loss = self.criterion(output, target_var)
             # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))
            losses.update(loss.data[0], data.size(0))
            top1.update(prec1[0], data.size(0))

        print(' * Prec@1 {0:.2f}'.format(float(top1.avg[0])))

        return top1.avg[0]

    def train_epoch(self):
        self.model.train()
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
                self.model.centers = self.model.centers.cuda()

            data_var, target_var = Variable(data), Variable(target)
            output = self.model(data_var)

            center_loss, self.model.centers = get_center_loss(self.model.centers, self.model.features, target_var, 1, self.model.num_classes)
            softmax_loss = self.criterion(output, target_var)
            loss = center_loss + softmax_loss

            # measure accuracy and record loss
            prec = accuracy(output.data, target, topk=(1,))
            self.losses.update(loss.data[0], data.size(0))
            self.top1.update(prec[0], data.size(0))

            # compute gradient and do SGD step
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if self.iteration >= self.max_iter:
                break

        print('Epoch: [{0}][{1}/{2}]\t'
        'Train Loss {3:.2f} ({4:.2f})\t'
        'Train Prec@1 {5:.2f} ({6:.2f})\t'.format(
        self.epoch, batch_idx, len(self.train_loader),
        float(self.losses.val), float(self.losses.avg), float(self.top1.val[0]), float(self.top1.avg[0])))
        self.losses.reset()
        self.top1.reset()

    def train(self):
        for epoch in itertools.count(self.epoch):
            self.epoch = epoch

            if self.val_loader:
                prec1 = self.validate()
            if self.iteration >= self.max_iter:
                break
            self.train_epoch()

            is_best = prec1 > self.best_prec1
            self.best_prec1 = max(prec1,self.best_prec1)
            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'centers': self.model.centers,
            }, is_best)
