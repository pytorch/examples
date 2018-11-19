import argparse
import os
import random
import shutil
import time
import warnings
import sys
import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--adam', dest='adam', action='store_true',
                    help='use adam optimizer')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--outdir', default='.', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--confusion', default=None, type=int,
                    metavar='N', help='generate confusion matrix against index on validation set')
parser.add_argument('--false-report', dest='false_report', action='store_true',
                    help='output false pos/neg report with confusion scores')
parser.add_argument('--train-report', dest='train_report', action='store_true',
                    help='report on the training set instead of the validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--disable-finetune', dest='disable_finetune', action='store_true',
                    help='disable training only on final layer')
parser.add_argument('--train-weights', default=None,
                    metavar='W', help='weight class labels during training (eg: 0=5,10=0.5)')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if not args.disable_finetune:
        num_classes = 1000 # TODO: maybe compute this based on args.data
        print("=> freezing {} weights except for last layer".format(args.arch))
        model = FineTuneModel(model, args.arch, num_classes)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.adam:
        print("Using adam optimizer with learning rate {}".format(args.lr))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        print("Using SGD optimizer with learning rate {}".format(args.lr))
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = ImageFolderWithPath(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    elif args.train_weights is not None:
        weights_list = args.train_weights.split(",")
        weights_lookup = {}
        for pair in weights_list:
            index, value = list(map(int, pair.split("=")))
            weights_lookup[index] = value
        train_weights = np.array([weights_lookup[s[1]] if s[1] in weights_lookup else 1.0 for s in train_dataset.imgs])
        print(train_weights[:1000])
        train_weights = torch.from_numpy(train_weights)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
        # sys.exit(0)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        ImageFolderWithPath(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.outdir != '.' and not os.path.exists(args.outdir):
      os.makedirs(args.outdir)

    if args.evaluate:
        validate(val_loader, model, criterion, args)

    if args.confusion is not None:
        if args.train_report:
            confusion(train_loader, model, criterion, args.confusion, args.false_report, args)
        else:
            confusion(val_loader, model, criterion, args.confusion, args.false_report, args)

    if args.evaluate or args.confusion is not None:
        # we are done
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.outdir)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, path) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, path) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def predict_wide(output, target, cindex):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    topk=(1,)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        np_pred = pred.cpu().data.numpy();
        np_targ = target.cpu().data.numpy();
        index_pred = np.squeeze((np_pred == cindex))
        index_targ = np.squeeze((np_targ == cindex))
        # print(index_pred.shape, index_targ.shape)
        # correct = torch.from_numpy(np.equal(index_pred, index_targ).astype(int))

        # correct_k = correct.float().sum(0, keepdim=True)
        # res = correct_k.mul_(100.0 / batch_size)
        return index_pred.tolist(), index_targ.tolist()


def confusion(val_loader, model, criterion, cindex, false_report, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        y_pred_c = []
        y_true_c = []
        false_pos_c = []
        false_neg_c = []
        for i, (input, target, path) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # add to confusion matrix
            # confusion_matrix.add(output.data.squeeze(), target.type(t.LongTensor))
            # print(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            y_pred_cindex, y_true_cindex = predict_wide(output, target, cindex)
            if false_report:
                for i in range(len(y_pred_cindex)):
                    if y_pred_cindex[i] and not y_true_cindex[i]:
                        false_pos_c.append(path[i])
                    elif not y_pred_cindex[i] and y_true_cindex[i]:
                        false_neg_c.append(path[i])

            # print(y_pred_cindex, y_true_cindex)
            y_pred_c = y_pred_c + y_pred_cindex
            y_true_c = y_true_c + y_true_cindex
            # conf_mat = confusion_matrix(y_true_cindex, y_pred_cindex)
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                # print(conf_mat)
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        conf_mat = metrics.confusion_matrix(y_true_c, y_pred_c)
        acc_sc = metrics.accuracy_score(y_true_c, y_pred_c)
        precision = metrics.precision_score(y_true_c, y_pred_c)
        recall = metrics.recall_score(y_true_c, y_pred_c)
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))
        print(" +[{}] accuracy {}, precision: {}, recall {}".format(cindex, acc_sc, precision, recall))
        print(conf_mat)
        if false_report:
            print("False Positives ({}):".format(len(false_pos_c)))
            for f in false_pos_c:
                print("FP {}".format(f))
            print("False Negatives ({}):".format(len(false_neg_c)))
            for f in false_neg_c:
                print("FN {}".format(f))

    return top1.avg


def save_checkpoint(state, is_best, outdir):
    filename='checkpoint.pth.tar'
    outfile1 = os.path.join(outdir, filename)
    torch.save(state, outfile1)
    if is_best:
        outfile2 = os.path.join(outdir, 'model_best.pth.tar')
        shutil.copyfile(outfile1, outfile2)


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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ImageFolderWithPath(datasets.ImageFolder):
    """
        Like ImageFolder, but also provides source path and blacklists
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=datasets.folder.default_loader):
        super(ImageFolderWithPath, self).__init__(root,
                                          transform=transform,
                                          target_transform=target_transform,
                                          loader=loader)

        # don't freak out if directories have mac "poopy files"
        candidates = [f for f in self.samples if not f[0].startswith(".")]
        if len(candidates) != len(self.samples):
            print("Info: removed {} files with bad filenames".format(len(self.samples) - len(candidates)))
        # remove blacklisted files (if any)
        blacklist_file = os.path.join(root, "blacklist.txt")
        if os.path.exists(blacklist_file):
            blacklist = list(line.strip() for line in open(blacklist_file))
            blacklist = [os.path.join(root, b) for b in blacklist if not b.startswith("#") and len(b) > 1]
            blacklist = set(blacklist)
            diagnostic_basename = os.path.basename(root)
            print("Info: {} found blacklist of length {}".format(diagnostic_basename, len(blacklist)))
            candidates2 = [f for f in candidates if not f[0] in blacklist]
            print("Info: {} removed {} files via blacklist".format(diagnostic_basename, len(candidates) - len(candidates2)))
            candidates = candidates2

        self.samples = candidates
        self.targets = [s[1] for s in candidates]


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


class FineTuneModel(nn.Module):
    """ Finetunes just the last layer

    This freezes the weights of all layers except the last one.
    You should also look into finetuning previous layers, but slowly
    Ideally, do this first, then unfreeze all layers and tune further with small lr

    Arguments:
        original_model: Model to finetune
        arch: Name of model architecture
        num_classes: Number of classes to tune for

    """
    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()
        self.dense_forward = False

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            self.features = original_model.features
            self.fc = nn.Sequential(*list(original_model.classifier.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(4096, num_classes)
            )
        elif arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(original_model.fc.in_features, num_classes)
            )
        elif arch.startswith('inception') :
            # inception is not working because:
            #  1) model.aux_logits = False is necessary: https://github.com/pytorch/vision/issues/302#issuecomment-341163548
            #  2) input size needs to be 299
            #  3) still something else is going on with finetune on features.forward:
            #     RuntimeError: size mismatch, m1: [32 x 277248], m2: [768 x 1000]
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, num_classes)
            )
        elif arch.startswith('densenet') :
            self.dense_forward = True

            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])

            # Get number of features of last layer
            num_feats = original_model.classifier.in_features

            # Plug our classifier
            self.classifier = nn.Sequential(
                nn.Linear(num_feats, num_classes)
            )

        else :
            raise("Finetuning not supported on this architecture yet. Feel free to add")

        self.unfreeze(False) # Freeze weights except last layer

    def unfreeze(self, unfreeze):
        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = unfreeze
        if hasattr(self, 'fc'):
            for p in self.fc.parameters():
                p.requires_grad = unfreeze

    def forward(self, x):
        f = self.features(x)
        if self.dense_forward:
            f = F.relu(f, inplace=True)
            f = F.avg_pool2d(f, kernel_size=7).view(f.size(0), -1)
        elif hasattr(self, 'fc'):
            f = f.view(f.size(0), -1)
            f = self.fc(f)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

if __name__ == '__main__':
    main()
