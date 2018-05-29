import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from dataset import Dataset, create_datasets, LFWPairedDataset
from loss import compute_center_loss, get_center_delta
from model import FaceModel
from device import device
from trainer import Trainer
from utils import download, generate_roc_curve
from metrics import compute_roc

parser = argparse.ArgumentParser(description='center loss example')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--log_dir', type=str,
                    help='log directory')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--resume', type=str,
                    help='model path to the resume training', default=False)
parser.add_argument('--dataset_dir', type=str,
                    help='directory with lfw dataset'
                         ' (default: $HOME/datasets/lfw)')
parser.add_argument('--weights', type=str, help='pretrained weights to load')
parser.add_argument('--evaluate', type=str,
                    help='evaluate specified model on lfw dataset')
parser.add_argument('--pairs', type=str,
                    help='path of pairs.txt '
                         '(default: $DATASET_DIR/pairs.txt)')
parser.add_argument('--roc', type=str,
                    help='path of roc.png to generated '
                         '(default: $DATASET_DIR/roc.png)')


def train():
    global args, dataset_dir, log_dir

    training_set, validation_set, num_classes = create_datasets(dataset_dir)
    train_transforms = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((96, 128)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
     )

    validation_transforms = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((96, 128)),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    training_dataset = Dataset(training_set, train_transforms)
    validation_dataset = Dataset(validation_set, validation_transforms)

    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=True
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=False
    )

    model = FaceModel(num_classes).to(device)
    if args.weights:
        model.load_state_dict(torch.load(args.weights), strict=False)

    trainables_wo_bn = [param for name, param in model.named_parameters() if
                        param.requires_grad and 'bn' not in name]
    trainables_only_bn = [param for name, param in model.named_parameters() if
                          param.requires_grad and 'bn' in name]

    optimizer = torch.optim.SGD([
        {'params': trainables_wo_bn, 'weight_decay': 0.0001},
        {'params': trainables_only_bn}
    ], lr=args.lr, momentum=0.9)

    trainer = Trainer(
        optimizer,
        model,
        training_dataloader,
        validation_dataloader,
        max_epoch=args.epochs,
        resume=args.resume,
        log_dir=log_dir
    )
    trainer.train()


def evaluate():
    global args, dataset_dir, log_dir

    pairs_path = args.pairs if args.pairs else \
        os.path.join(dataset_dir, 'pairs.txt')

    if not os.path.isfile(pairs_path):
        download(dataset_dir, 'http://vis-www.cs.umass.edu/lfw/pairs.txt')

    eval_transforms = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((96, 128)),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
     )

    dataset = LFWPairedDataset(dataset_dir, pairs_path, eval_transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    model = FaceModel().to(device)

    checkpoint = torch.load(args.evaluate)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    embedings_a = torch.zeros(len(dataset), 512)
    embedings_b = torch.zeros(len(dataset), 512)
    matches = torch.zeros(len(dataset), dtype=torch.uint8)

    for iteration, (images_a, images_b, batched_matches) \
            in enumerate(dataloader):
        current_batch_size = len(batched_matches)
        images_a = images_a.to(device)
        images_b = images_b.to(device)

        _, batched_embedings_a = model(images_a)
        _, batched_embedings_b = model(images_b)

        start = args.batch_size * iteration
        end = start + current_batch_size

        embedings_a[start:end, :] = batched_embedings_a.data
        embedings_b[start:end, :] = batched_embedings_b.data
        matches[start:end] = batched_matches.data

    thresholds = np.arange(0, 4, 0.1)
    distances = torch.sum(torch.pow(embedings_a - embedings_b, 2), dim=1)

    tpr, fpr, accuracy = compute_roc(
        distances,
        matches,
        thresholds
    )

    roc_file = args.roc if args.roc else os.path.join(log_dir, 'roc.png')
    generate_roc_curve(fpr, tpr, roc_file)
    print('Model accuracy is {}'.format(accuracy))
    print('ROC curve generated at {}'.format(roc_file))


if __name__ == '__main__':
    home = os.path.expanduser("~")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir if args.dataset_dir else os.path.join(
        home, 'datasets', 'lfw')
    log_dir = args.log_dir if args.log_dir else os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'logs')
    evaluate() if args.evaluate else train()
