import argparse
import datetime
import logging
import os
import random
import shutil
import time
import warnings
from enum import Enum
from typing import get_type_hints, Tuple, List, Union, Dict, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassPrecisionRecallCurve

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


# ================
# Model evaluation
# ================

class TrainMetrics(NamedTuple):
    class_labels: List[int]
    acc_balanced: float
    f1_micro: float
    f1_macro: float
    prec_micro: float
    prec_macro: float
    rec_micro: float
    rec_macro: float


class ValidationMetrics(NamedTuple):
    class_labels: List[int]
    acc_balanced: float
    f1_micro: float
    f1_macro: float
    prec_micro: float
    prec_macro: float
    rec_micro: float
    rec_macro: float
    f1_per_class: List[Tuple[int, float]]
    conf_matrix: np.array
    labels_true: np.array
    labels_pred: np.array
    labels_probs: np.array
    fig_pr_curve_micro: plt.Figure


class EarlyStopping:
    """
    Based on:
    - https://pytorch.org/ignite/_modules/ignite/handlers/early_stopping.html#EarlyStopping
    - https://github.com/Bjarten/early-stopping-pytorch
    """

    def __init__(self, patience: int = 3, min_delta: float = 1, min_epochs: int = 50):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.epoch_min_validation_loss = 0
        self.should_stop = False
        self.min_epochs = min_epochs

    def __call__(self, validation_loss, epoch):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.epoch_min_validation_loss = epoch
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience and epoch >= self.min_epochs:
                self.should_stop = True


# ================
# Custom model & data
# ================
def safe_import(module_name):
    import importlib
    import sys

    if module_name in sys.modules:
        return sys.modules[module_name]

    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError as e:
        print(f'Error importing module {module_name}. Make sure the module exists and can be imported.')
        raise e


def get_module_method(module_name, method_name, expected_type_hint):
    if hasattr(module_name, method_name) and callable(getattr(module_name, method_name)):
        method = getattr(module_name, method_name)
        if not get_type_hints(method)['return'] == expected_type_hint:
            raise Exception(
                f'The provided method {module_name}.{method_name} does not respect the '
                f'expected type hint {expected_type_hint}')
        return method()
    else:
        raise Exception(f'The provided module {module_name} does not have method {method_name}')


def get_run_name(model, train_dataset, val_dataset, args):
    today = datetime.datetime.now().strftime('%m%d-%H%M')

    model_info = model.__class__.__name__
    dataset_info = train_dataset.__class__.__name__
    if args.use_module_definitions:
        module = safe_import(args.use_module_definitions.replace('.py', ''))
        try:
            model_info = get_module_method(module, 'get_model_info', str)
        except:
            pass
        try:
            dataset_info = get_module_method(module, 'get_dataset_info', str)
        except:
            pass

    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)

    return (f"{today}_{model_info}"
            f"_{dataset_info}-train-{train_dataset_size}-val-{val_dataset_size}")


# ================
# Arguments
# ================

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-m', '--use-module-definitions', metavar='MODULE', default=None,
                    help='load a custom py file for the model and/or dataset & loader.'
                         'The file can contain the following functions: '
                         'get_model() -> nn.Module'
                         'get_train_dataset() -> torch.utils.data.Dataset'
                         'get_val_dataset() -> torch.utils.data.Dataset'
                         'get_train_loader() -> torch.utils.data.DataLoader'
                         'get_val_loader() -> torch.utils.data.DataLoader'
                         '(default: None)')
parser.add_argument('-tb', '--tb-summary-writer-dir', metavar='SUMMARY_DIR', default=None)
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
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-ch', '--checkpoints', default='', type=str,
                    help='path to checkpoints dir (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")

log = print
best_acc1 = 0
best_metrics = ValidationMetrics([], 0, 0, 0, 0, 0, 0, 0, [], [], [], [], [], None)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn(
                "nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        ngpus_per_node = 1

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
    global best_acc1, best_metrics, log
    args.gpu = gpu

    if args.use_module_definitions:
        module = safe_import(args.use_module_definitions.replace('.py', ''))
        try:
            logger = get_module_method(module, 'get_logger', logging.Logger)
            log = logger.info
        except:
            pass

    if args.gpu is not None:
        log("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        log("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        if not args.use_module_definitions:
            log("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()
        else:
            module = safe_import(args.use_module_definitions.replace('.py', ''))
            model = get_module_method(module, 'get_model', nn.Module)

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        log('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            log("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            log("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            log("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    if args.dummy:
        log("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        if not args.use_module_definitions:
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
        else:
            module = safe_import(args.use_module_definitions.replace('.py', ''))
            train_dataset = get_module_method(module, 'get_train_dataset', torch.utils.data.Dataset)
            val_dataset = get_module_method(module, 'get_val_dataset', torch.utils.data.Dataset)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    if not args.use_module_definitions:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    else:
        module = safe_import(args.use_module_definitions.replace('.py', ''))
        train_loader = get_module_method(module, 'get_train_loader', torch.utils.data.DataLoader)
        val_loader = get_module_method(module, 'get_val_loader', torch.utils.data.DataLoader)

    target_class_translations = None
    if args.use_module_definitions:
        try:
            module = safe_import(args.use_module_definitions.replace('.py', ''))
            target_class_translations = get_module_method(module, 'target_class_translations', Dict[int, str])
            log(f'Loaded target_class_translations from {args.use_module_definitions}')
        except Exception as e:
            log(f'Error getting target_class_translations from {args.use_module_definitions}: {e}')

    def get_target_class(cl: int) -> str:
        if target_class_translations:
            return target_class_translations[cl]
        return f"Class-{cl}"

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    run_name = get_run_name(model, train_dataset, val_dataset, args)
    tensorboard_writer = None
    if args.tb_summary_writer_dir:
        tb_log_dir_path = os.path.join(args.tb_summary_writer_dir, run_name)
        tensorboard_writer = SummaryWriter(tb_log_dir_path)
        log(f'TensorBoard summary writer is created at {tb_log_dir_path}')

        try:
            model.eval()
            with torch.no_grad():
                images, _ = next(iter(train_loader))
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                tensorboard_writer.add_graph(model, images)
        except Exception as e:
            log(f"Failed to add graph to tensorboard.")

    early_stopping = EarlyStopping(patience=5, min_delta=0.5, min_epochs=50)
    try:
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            # train for one epoch
            train_acc1, train_loss, train_metrics = train(train_loader, model, criterion, optimizer, epoch, device,
                                                          args)

            # evaluate on validation set
            val_acc1, val_loss, val_metrics = validate(val_loader, model, criterion, args)
            scheduler.step()
            early_stopping(val_loss, epoch)

            # remember best acc@1 and save checkpoint
            is_best = val_acc1 > best_acc1
            best_acc1 = max(val_acc1, best_acc1)
            best_metrics = val_metrics if val_metrics.f1_micro > best_metrics.f1_micro else best_metrics

            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0) or \
                    epoch == args.epochs - 1:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, is_best, run_name, args.checkpoints)

            if tensorboard_writer:
                tensorboard_writer.add_scalars('Loss', dict(train=train_loss, val=val_loss), epoch + 1)
                tensorboard_writer.add_scalars('Metrics/Accuracy',
                                               dict(val_acc=val_acc1 / 100.0,
                                                    val_bacc=val_metrics.acc_balanced,
                                                    train_acc=train_acc1 / 100.0,
                                                    train_bacc=train_metrics.acc_balanced),
                                               epoch + 1)
                tensorboard_writer.add_scalars('Metrics/F1',
                                               dict(val_micro=val_metrics.f1_micro,
                                                    val_macro=val_metrics.f1_macro,
                                                    train_micro=train_metrics.f1_micro,
                                                    train_macro=train_metrics.f1_macro),
                                               epoch + 1)
                tensorboard_writer.add_scalars('Metrics/Precision',
                                               dict(val_micro=val_metrics.prec_micro,
                                                    val_macro=val_metrics.prec_macro,
                                                    train_micro=train_metrics.prec_micro,
                                                    train_macro=train_metrics.prec_macro),
                                               epoch + 1)
                tensorboard_writer.add_scalars('Metrics/Recall',
                                               dict(val_micro=val_metrics.rec_micro,
                                                    val_macro=val_metrics.rec_macro,
                                                    train_micro=train_metrics.rec_micro,
                                                    train_macro=train_metrics.rec_macro),
                                               epoch + 1)
                tensorboard_writer.add_scalars('Metrics/F1/class',
                                               {get_target_class(cl): f1 for cl, f1 in val_metrics.f1_per_class},
                                               epoch + 1)

                if epoch < 10 or epoch % 5 == 0 or epoch == args.epochs - 1:
                    class_names = [get_target_class(cl) for cl in list({l for l in val_metrics.class_labels})]
                    fig_abs, _ = plot_confusion_matrix(val_metrics.conf_matrix, class_names=class_names,
                                                       normalize=False)
                    fig_rel, _ = plot_confusion_matrix(val_metrics.conf_matrix, class_names=class_names, normalize=True)
                    tensorboard_writer.add_figure('Confusion matrix', fig_abs, epoch + 1)
                    tensorboard_writer.add_figure('Confusion matrix/normalized', fig_rel, epoch + 1)

                    for cl in val_metrics.class_labels:
                        class_index = int(cl)
                        labels_true = val_metrics.labels_true == class_index
                        pred_probs = val_metrics.labels_probs[:, class_index]
                        tensorboard_writer.add_pr_curve(f'PR curve/{get_target_class(class_index)}',
                                                        labels_true, pred_probs, epoch + 1)

                    tensorboard_writer.add_figure('PR curve', val_metrics.fig_pr_curve_micro, epoch + 1)

            if early_stopping.should_stop:
                log(f"Early stopping at epoch {epoch + 1}")
                break


    except KeyboardInterrupt:
        log('Training interrupted, saving hparams to TensorBoard...')
    finally:
        if args.use_module_definitions:
            module = safe_import(args.use_module_definitions.replace('.py', ''))
            hparams = get_module_method(module, 'get_hparams', Dict[str, Union[int, float, bool, str]])
            if tensorboard_writer and hparams:
                tensorboard_writer.add_hparams(hparams, {
                    'hparams/Accuracy': best_acc1 / 100.0,
                    'hparams/F1-micro': best_metrics.f1_micro,
                    'hparams/F1-macro': best_metrics.f1_macro,
                    'hparams/P-micro': best_metrics.prec_micro,
                    'hparams/P-macro': best_metrics.prec_macro,
                    'hparams/R-micro': best_metrics.rec_micro,
                    'hparams/R-macro': best_metrics.rec_macro,
                })


def train(train_loader, model, criterion, optimizer, epoch, device, args) -> Tuple[float, float, TrainMetrics]:
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc_top1 = AverageMeter('Acc@1', ':6.2f')
    acc_top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc_top1, acc_top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    # for train metrics
    labels_true = np.array([], dtype=np.int64)
    labels_pred = np.array([], dtype=np.int64)
    labels_probs = []

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        acc_top1.update(acc1[0], images.size(0))
        acc_top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predicted_values, predicted_indices = torch.max(output.data, 1)
            labels_true = np.append(labels_true, target.cpu().numpy())
            labels_pred = np.append(labels_pred, predicted_indices.cpu().numpy())

            class_probs_batch = [F.softmax(el, dim=0) for el in output]
            labels_probs.append(class_probs_batch)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    if args.distributed:
        acc_top1.all_reduce()
        acc_top5.all_reduce()

    labels_probs = torch.cat([torch.stack(batch) for batch in labels_probs]).cpu()
    metrics = calculate_train_metrics(labels_true, labels_pred, labels_probs)

    return acc_top1.avg, loss.item(), metrics


def validate(val_loader, model, criterion, args) -> Tuple[float, float, "ValidationMetrics"]:
    def run_validate(loader, base_progress=0) -> ValidationMetrics:
        labels_true = np.array([], dtype=np.int64)
        labels_pred = np.array([], dtype=np.int64)
        labels_probs = []

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                acc_top1.update(acc1[0], images.size(0))
                acc_top5.update(acc5[0], images.size(0))

                # measure f1, precision, recall
                with torch.no_grad():
                    predicted_values, predicted_indices = torch.max(output.data, 1)
                    labels_true = np.append(labels_true, target.cpu().numpy())
                    labels_pred = np.append(labels_pred, predicted_indices.cpu().numpy())

                    class_probs_batch = [F.softmax(el, dim=0) for el in output]
                    labels_probs.append(class_probs_batch)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

        labels_probs = torch.cat([torch.stack(batch) for batch in labels_probs]).cpu()

        return calculate_validation_metrics(labels_true, labels_pred, labels_probs)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    acc_top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    acc_top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, acc_top1, acc_top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    metrics = run_validate(val_loader)
    if args.distributed:
        acc_top1.all_reduce()
        acc_top5.all_reduce()
        losses.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        metrics = run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return acc_top1.avg, losses.avg, metrics


def save_checkpoint(state, is_best, run_info: str = "", dir="./"):
    filename = f'{run_info}_checkpoint.pth.tar'
    filepath = os.path.join(dir, filename)
    log(f'Saving checkpoint to {filename} at {filepath}')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, filepath.replace("checkpoint", "model_best"))


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    val: float
    sum: float
    count: int
    avg: float

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters: List[AverageMeter], prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        log('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        log(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# =================
# Metrics
# =================


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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def calculate_train_metrics(labels_true: np.array, labels_pred: np.array,
                            labels_probs: torch.Tensor) -> TrainMetrics:
    unique_labels = list({l for l in labels_true})
    f1_micro = f1_score(labels_true, labels_pred, average="micro")
    f1_macro = f1_score(labels_true, labels_pred, average="macro")

    acc_balanced = balanced_accuracy_score(labels_true, labels_pred)
    prec_micro = precision_score(labels_true, labels_pred, average="micro")
    prec_macro = precision_score(labels_true, labels_pred, average="macro")
    rec_micro = recall_score(labels_true, labels_pred, average="micro")
    rec_macro = recall_score(labels_true, labels_pred, average="macro")

    return TrainMetrics(
        unique_labels,
        acc_balanced,
        f1_micro, f1_macro,
        prec_micro, prec_macro,
        rec_micro, rec_macro
    )


def calculate_validation_metrics(labels_true: np.array, labels_pred: np.array,
                                 labels_probs: torch.Tensor) -> ValidationMetrics:
    unique_labels = list({l for l in labels_true})
    f1_per_class = f1_score(labels_true, labels_pred, average=None, labels=unique_labels)
    f1_micro = f1_score(labels_true, labels_pred, average="micro")
    f1_macro = f1_score(labels_true, labels_pred, average="macro")

    acc_balanced = balanced_accuracy_score(labels_true, labels_pred)
    prec_micro = precision_score(labels_true, labels_pred, average="micro")
    prec_macro = precision_score(labels_true, labels_pred, average="macro")
    rec_micro = recall_score(labels_true, labels_pred, average="micro")
    rec_macro = recall_score(labels_true, labels_pred, average="macro")

    conf_matrix = confusion_matrix(labels_true, labels_pred)

    fig_pr_curve_micro, _ = plot_pr_curve_micro(len(unique_labels), labels_probs, torch.tensor(labels_true))

    return ValidationMetrics(
        unique_labels,
        acc_balanced,
        f1_micro, f1_macro,
        prec_micro, prec_macro,
        rec_micro, rec_macro,
        [(cl, f1) for cl, f1 in zip(unique_labels, f1_per_class)],
        conf_matrix,
        labels_true,
        labels_pred,
        labels_probs,
        fig_pr_curve_micro
    )


def plot_confusion_matrix(cm, class_names, normalize=False):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(10, 10))

    if normalize:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        colormap = "Greens"
    else:
        colormap = "Blues"

    plt.imshow(cm, interpolation='nearest', cmap=colormap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig, plt


def plot_pr_curve_micro(num_classes: int, labels_probs: torch.Tensor, labels_true: torch.Tensor):
    fig = plt.figure(figsize=(8, 8))
    metric = MulticlassPrecisionRecallCurve(num_classes=num_classes, average="micro")
    metric.update(labels_probs, labels_true)
    metric.plot(ax=plt.gca())
    plt.title("PR curve micro avg.")
    plt.tight_layout()
    return fig, plt


if __name__ == '__main__':
    main()
