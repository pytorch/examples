from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable

import net as models
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch Fast Neural Style Generate Example')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--model', default='model.pth', type=str,
                    help='the path of the model file')
parser.add_argument('--input', type=str, required=True,
                    help='the path of the input image')
parser.add_argument('--output', default='output.jpg', type=str,
                    help='the path of the output image')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

model = models.FastStyleNet()
model.load_state_dict(torch.load(args.model))

# load image
img = Image.open(args.input)
img = np.array(img)  # PIL->numpy
img = np.array(img[..., ::-1])  # RGB->BGR
img = img.transpose(2, 0, 1)  # HWC->CHW
img = img.reshape((1, ) + img.shape)  # CHW->BCHW
img = torch.from_numpy(img).float()
img = Variable(img, volatile=True)

if args.cuda:
    model.cuda()
    img = img.cuda()

model.eval()
output = model(img)

# save output
output = output.data.cpu().clamp(0, 255).byte().numpy()
output = output[0].transpose((1, 2, 0))
output = output[..., ::-1]
output = Image.fromarray(output)
output.save(args.output)
