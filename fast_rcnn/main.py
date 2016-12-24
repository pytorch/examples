import argparse
import time

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

import torch.optim as optim

from voc import VOCDetection, TransformVOCDetectionAnnotation

import importlib

#from model import model

from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch Faster R-CNN Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', '-m', metavar='MODEL', default='model',
                    help='file containing model definition '
                    '(default: model)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

cls = ('__background__', # always index 0
       'aeroplane', 'bicycle', 'bird', 'boat',
       'bottle', 'bus', 'car', 'cat', 'chair',
       'cow', 'diningtable', 'dog', 'horse',
       'motorbike', 'person', 'pottedplant',
       'sheep', 'sofa', 'train', 'tvmonitor')
class_to_ind = dict(zip(cls, range(len(cls))))

args = parser.parse_args()
model = importlib.import_module(args.model).model

train = VOCDetection(args.data, 'train',
            transform=transforms.ToTensor(),
            target_transform=TransformVOCDetectionAnnotation(class_to_ind, False))

def collate_fn(batch):
    imgs, gt = zip(*batch)
    return imgs[0].unsqueeze(0), gt[0]

train_loader = torch.utils.data.DataLoader(
            train, batch_size=1, shuffle=True,
            num_workers=0, collate_fn=collate_fn)



optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)

def train(train_loader, model, optimizer, epoch):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()

  model.train()
  end = time.time()
  for i, (im, gt) in (enumerate(train_loader)):
    # measure data loading time
    data_time.update(time.time() - end)

    optimizer.zero_grad()
    loss, scores, boxes = model((im, gt))
    loss.backward()
    optimizer.step()
    
    losses.update(loss.data[0], im.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()
    if i % args.print_freq == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            #'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
            .format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses,
            #top1=top1, top5=top5
            ))


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

for epoch in range(0, 10):
  train(train_loader, model, optimizer, epoch)

#from IPython import embed; embed()

#if __name__ == '__main__':
#  main()
