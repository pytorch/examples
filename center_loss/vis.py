from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import datasets, transforms
from face_model import FaceModel, Net
from data import ImageDataset
import trainer as tn
import math
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

parser = argparse.ArgumentParser(description='PyTorch face recognition Example')
parser.add_argument('--test_batch_size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--root', type=str,
        help='path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='./datasets/casia_mtcnnpy_160')
parser.add_argument('--resume', type=str,
        help='model path to the resume training',
        default='')

def visual_feature_space(features, labels, num_classes, name_dict):
    num = len(labels)

    title_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space
    axis_font = {'fontname':'Arial', 'size':'20'}

    # draw
    palette = np.array(sns.color_palette("hls", num_classes))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(features[:,0], features[:,1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    # ax.axis('off')
    # ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(num_classes):
        # Position of each label.
        xtext, ytext = np.median(features[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, name_dict[i])
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    ax.set_xlabel('Activation of the 1st neuron', **axis_font)
    ax.set_ylabel('Activation of the 2nd neuron', **axis_font)
    ax.set_title('softmax_loss + center_loss', **title_font)
    ax.set_axis_bgcolor('grey')
    f.savefig('center_loss.png')
    plt.show()
    return f, ax, sc, txts

def validation_iterator(dataLoader):
    for data, target in dataLoader:
        yield data, target

def main():
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    # 1. dataset
    root = args.root
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    # test_transforms = transforms.Compose([transforms.Scale((96,128)),
    #                                       transforms.ToTensor(),
    #                                       transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
    # test_dataset = ImageDataset(root, transform=test_transforms, train=False)
    # test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=args.test_batch_size, shuffle=True, **kwargs)
    val_iterator = validation_iterator(test_loader)

    # 2. model
    print('construct model')
    model = Net(10)
    if cuda:
        model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            model.load_state_dict(checkpoint['state_dict'])
            model.centers = checkpoint['centers']
            print("=> loaded checkpoint '{}'".format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # extract feature
    print('extracting feature')
    embeds = []
    labels = []
    for data, target in val_iterator:
        if cuda:
            data, target = data.cuda(), target.cuda(async=True)
            model.centers = model.centers.cuda()
        data_var = Variable(data, volatile=True)
        target_var = Variable(target, volatile=True)
        target_var = Variable(target, volatile=True)
        # compute output
        output = model(data_var)
        feature = model.features

        embeds.append( feature.data.cpu().numpy() )
        labels.append( target.cpu().numpy() )


    embeds = np.vstack(embeds)
    labels = np.hstack(labels)

    print('embeds shape is ', embeds.shape)
    print('labels shape is ', labels.shape)

    # prepare dict for display
    namedict = dict()
    for i in range(10):
        namedict[i]=str(i)

    visual_feature_space(embeds, labels, 10, namedict)

if __name__ == '__main__':
    main()
