from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import net as models
import numpy as np
import os
import cv2

def load_image(path):
    img = cv2.imread(path).astype('float32')
    img = img.transpose(2, 0, 1)
    return img

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--model', default='model.pth', type=str)
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', default='output.jpg', type=str)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

model = models.FastStyleNet()
model.load_state_dict(torch.load(args.model))

img = load_image(args.input)
img = img.reshape((1,)+img.shape)
img = torch.from_numpy(img)
img = Variable(img, volatile=True)

if(args.cuda):
    model.cuda()
    img = img.cuda()

model.eval()
result = model(img)
result = result.data.numpy()
result = result[0].transpose((1, 2, 0))
cv2.imwrite(args.output,result)




