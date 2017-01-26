from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.serialization import load_lua

import net as models
import numpy as np
import os
from datetime import datetime
import cv2

def download_convert_vgg16_model():
    if not os.path.exists('vgg16feature.pth'):
        if not os.path.exists('vgg16.t7'):
            os.system('wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7')
        vgglua = load_lua('vgg16.t7')
        vgg = models.vgg16feature()
        for src,dst in zip(vgglua.parameters()[0],vgg.parameters()):
            dst[:] = src[:]
        torch.save(vgg.state_dict(), 'vgg16feature.pth')

def load_image(path, size=None):
    img = cv2.imread(path).astype('float32')
    img = cv2.resize(img,(size,size))
    img = img.transpose(2, 0, 1)
    return img

def gram_matrix(y):
    b, ch, h, w = y.size()
    features = y.view(b, ch, w*h)
    features_t = features.transpose(1,2)
    gram = features.bmm(features_t)/(ch*h*w)
    return gram

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Fast Neural Style Example')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--dataset', '-d', default='../../coco5000', type=str,
                    help='dataset directory path (according to the paper, use MSCOCO 80k images)')
parser.add_argument('--style_image', '-s', type=str, default='starry_night.jpg',
                    help='style image path')
parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--lambda_feat', default=1.0, type=float)
parser.add_argument('--lambda_style', default=5.0, type=float)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

lambda_f = args.lambda_feat
lambda_s = args.lambda_style
image_size = args.image_size
n_epoch = args.epochs

fs = os.listdir(args.dataset)
fs.sort()
imagepaths = []
for fn in fs:
    base, ext = os.path.splitext(fn)
    if ext == '.jpg' or ext == '.png':
        imagepath = os.path.join(args.dataset,fn)
        imagepaths.append(imagepath)
n_iter = len(imagepaths)


download_convert_vgg16_model()
model = models.FastStyleNet()
vgg = models.VGGFeature()
vgg.load_state_dict(torch.load('vgg16feature.pth'))
optimizer = optim.Adam(model.parameters(), lr=args.lr)
if args.cuda:
    model.cuda()
    vgg.cuda()


style = load_image(args.style_image,image_size) - 120
style = style.reshape((1,)+style.shape)
style = torch.from_numpy(style)
if(args.cuda):
    style = style.cuda()
feature_s = vgg(Variable(style, volatile=True))
gram_s = [gram_matrix(y) for y in feature_s]

loss_fn = torch.nn.MSELoss()
model.train()
for epoch in range(n_epoch):
    print('epoch', epoch)
    for i in range(n_iter):
        optimizer.zero_grad()

        img = load_image(imagepaths[i], image_size)
        img = img.reshape((1,)+img.shape)
        x = torch.from_numpy(img)
        if(args.cuda):
            x = x.cuda()

        xc = Variable(x.clone(), volatile=True)
        x = Variable(x)
        y = model(x)

        xc -= 120
        y -= 120
        feature = vgg(xc)
        feature_hat = vgg(y)

        L_feat = lambda_f * loss_fn(feature_hat[2], Variable(feature[2].data,requires_grad=False)) #relu3_3
        L_sytle = 0.0
        for m in range(0,len(feature_hat)):
            L_sytle += lambda_s * loss_fn(gram_matrix(feature_hat[m]), Variable(gram_s[m].data,requires_grad=False))

        L = L_feat + L_sytle
        L.backward()
        optimizer.step()
        
        dt = datetime.now().strftime('%H:%M:%S')
        print('{} epoch {} batch {}/{}    loss is {}'.format(dt,epoch, i, n_iter, int(L.data[0])))

    torch.save(model.state_dict(), 'model_{}.pth'.format(epoch))
torch.save(model.state_dict(), 'model.pth')

