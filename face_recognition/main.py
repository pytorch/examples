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
from utils import download, generate_roc_curve, image_loader
from metrics import compute_roc, select_threshold
from imageaug import transform_for_infer, transform_for_training

parser = argparse.ArgumentParser(description='center loss example')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--log_dir', type=str,
                    help='log directory')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--resume', type=str,
                    help='model path to the resume training', default=False)
parser.add_argument('--dataset_dir', type=str,
                    help='directory with lfw dataset'
                         ' (default: $HOME/datasets/lfw)')
parser.add_argument('--weights', type=str, help='pretrained weights to load '
                    'default: ($LOG_DIR/resnet18.pth)')
parser.add_argument('--evaluate', type=str,
                    help='evaluate specified model on lfw dataset')
parser.add_argument('--pairs', type=str,
                    help='path of pairs.txt '
                         '(default: $DATASET_DIR/pairs.txt)')
parser.add_argument('--roc', type=str,
                    help='path of roc.png to generated '
                         '(default: $DATASET_DIR/roc.png)')
parser.add_argument('--verify-model', type=str,
                    help='verify 2 images of face belong to one person,'
                         'the param is the model to use')
parser.add_argument('--verify-images', type=str,
                    help='verify 2 images of face belong to one person,'
                         'split image pathes by comma')

RESNET18_WEIGHTS = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'


def train():
    global args, dataset_dir, log_dir

    training_set, validation_set, num_classes = create_datasets(dataset_dir)

    training_dataset = Dataset(training_set, transform_for_training())
    validation_dataset = Dataset(validation_set, transform_for_infer())

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
    if not args.resume:
        weights = args.weights if args.weights else download(log_dir,
                RESNET18_WEIGHTS, 'resnet18.pth')
        model.load_state_dict(torch.load(weights), strict=False)
        print('weights loaded from {}'.format(weights))

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

    dataset = LFWPairedDataset(dataset_dir, pairs_path, transform_for_infer())
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

def verify():
    global args, dataset_dir, log_dir

    model = FaceModel().to(device)
    checkpoint = torch.load(args.verify_model)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    image_a, image_b = args.verify_images.split(',')
    image_a = transform_for_infer()(image_loader(image_a))
    image_b = transform_for_infer()(image_loader(image_b))
    images = torch.stack([image_a, image_b]).to(device)

    _, (embedings_a, embedings_b) = model(images)

    distance = torch.sum(torch.pow(embedings_a - embedings_b, 2)).item()
    print("distance: {}".format(distance))


if __name__ == '__main__':
    home = os.path.expanduser("~")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir if args.dataset_dir else os.path.join(
        home, 'datasets', 'lfw')
    log_dir = args.log_dir if args.log_dir else os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'logs')

    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    if args.evaluate:
        evaluate()
    elif args.verify_model:
        verify()
    else:
        train()