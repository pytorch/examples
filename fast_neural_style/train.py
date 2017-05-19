from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.serialization import load_lua
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import net as models
import numpy as np
import os
import math
from datetime import datetime
from PIL import Image


def download_convert_vgg16_model():
    if not os.path.exists('vgg16feature.pth'):
        if not os.path.exists('vgg16.t7'):
            os.system('wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7')
        vgglua = load_lua('vgg16.t7')
        vgg = models.VGGFeature()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst[:] = src[:]
        torch.save(vgg.state_dict(), 'vgg16feature.pth')


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


# batch: Bx3xHxW BGR [0,255] Variable
def vgg_preprocessing(batch):
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch -= Variable(mean)


def save_model(model, filename):
    state = model.state_dict()
    for key in state:
        state[key] = state[key].clone().cpu()
    torch.save(state, filename)


# tensor: RGB CxHxW [0,255]
def tensor_save_rgbimage(tensor, filename):
    img = tensor.clone().cpu().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename)


# result: RGB CxHxW [0,255] torch.FloatTensor
def tensor_load_rgbimage(filename, size=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


# batch: Bx3xHxW
def batch_rgb_to_bgr(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch


def batch_bgr_to_rgb(batch):
    return batch_rgb_to_bgr(batch)


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Fast Neural Style Example')
parser.add_argument('--epochs', default=2, metavar='N', type=int,
                    help='number of epochs to train (default: 2)')
parser.add_argument('--lr', default=0.001, metavar='LR', type=float,
                    help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--dataset', default='../../dataset', type=str,
                    help='dataset directory path')
parser.add_argument('--style_image', default='images/wave.jpg', type=str,
                    help='style image path')
parser.add_argument('--prefix', type=str, default='modelprefix',
                    help='save model with the prefix')
parser.add_argument('--batch_size', default=1, type=int,
                    help='batch size')
parser.add_argument('--checkpoint', default=0, type=int,
                    help='save model every checkpoint')
parser.add_argument('--image_size', default=256, type=int,
                    help='resize the content image in dataset')
parser.add_argument('--style_size', default=256, type=int,
                    help='resize the style image')
parser.add_argument('--lambda_feat', default=1.0, type=float)
parser.add_argument('--lambda_style', default=5.0, type=float)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

transform = transforms.Compose([transforms.Scale(args.image_size),
                               transforms.CenterCrop(args.image_size),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: x.mul(255))])
train = datasets.ImageFolder(args.dataset, transform)
train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=4)
n_iter = len(train_loader)

download_convert_vgg16_model()
model = models.FastStyleNet()
vgg = models.VGGFeature()
vgg.load_state_dict(torch.load('vgg16feature.pth'))
optimizer = optim.Adam(model.parameters(), lr=args.lr)
if args.cuda:
    model.cuda()
    vgg.cuda()

style = tensor_load_rgbimage(args.style_image, args.style_size)
style = style.repeat(args.batch_size, 1, 1, 1)
style = batch_rgb_to_bgr(style)

if args.cuda:
    style = style.cuda()
style_var = Variable(style, volatile=True)
vgg_preprocessing(style_var)
feature_s = vgg(style_var)
gram_s = [gram_matrix(y) for y in feature_s]

it = 0
loss_fn = torch.nn.MSELoss()
model.train()
for epoch in range(args.epochs):
    for batch in train_loader:
        optimizer.zero_grad()

        # batch[0] is RGB [0,1] BxCxHxW
        # data is BGR [0,255] BxCxHxW
        # y is BGR [0,255] BxCxHxW
        data = batch[0].clone()
        data = batch_rgb_to_bgr(data)
        if args.cuda:
            data = data.cuda()

        x = Variable(data.clone())
        y = model(x)
        vgg_preprocessing(y)
        feature_hat = vgg(y)

        xc = Variable(data.clone(), volatile=True)
        vgg_preprocessing(xc)
        feature = vgg(xc)

        feature_v = Variable(feature[1].data, requires_grad=False)
        L = args.lambda_feat * loss_fn(feature_hat[1], feature_v)
        for m in range(0, len(feature_hat)):
            gram_v = Variable(gram_s[m].data, requires_grad=False)
            L += args.lambda_style * loss_fn(gram_matrix(feature_hat[m]), gram_v)
        L.backward()
        optimizer.step()

        dt = datetime.now().strftime('%H:%M:%S')
        print('{} epoch {} batch {}/{}    loss is {}'.format(dt, epoch, it, n_iter, L.data[0]))

        if args.checkpoint > 0 and 0 == it % args.checkpoint:
            save_model(model, '{}_{}_{}.pth'.format(args.prefix, epoch, it))
        it = it + 1
    save_model(model, '{}_{}.pth'.format(args.prefix, epoch))
save_model(model, '{}.pth'.format(args.prefix))
