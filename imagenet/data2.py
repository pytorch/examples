import math
import random

import torch

import torch.utils.data as data

import os.path
import torchfile
import numpy
from PIL import Image

###########################################################
# This is the main imagenet loading logic
###########################################################
class ImagenetDataset(data.Dataset):
    
    def __init__(self, path, jitter):
        self.path = path 
        self.data = torchfile.load(path)
        self.jitter = jitter
        self.res = 256

    def __len__(self):
        # return 1000
        return len(self.data.imagePath)

    def __getitem__(self, i):
        imagePath = self.data.imagePath[i].tobytes()
        try:
            # remove the null-terminators
            imagePath = imagePath[:imagePath.index('\0')]
        except:
            pass
        pic = Image.open(imagePath)
        pic = pic.convert('RGB')
        if pic.size[0] > pic.size[1]:
            pic.resize((self.res * pic.size[0]/pic.size[1], self.res), Image.BILINEAR)
        else:
            pic.resize((self.res, self.res * pic.size[1]/pic.size[0]), Image.BILINEAR)
        
        h1 = None
        w1 = None
        if self.jitter:
            # random crop
            h1 = math.ceil(random.uniform(1e-2, pic.size[0] - self.res))
            w1 = math.ceil(random.uniform(1e-2, pic.size[1] - self.res))
        else:
            # center crop
            w1 = math.ceil(pic.size[0] - self.res)/2
            h1 = math.ceil(img.size[1] - self.res)/2

        pic = pic.crop((w1, h1, w1 + self.res, h1 + self.res))

        if self.jitter and random.uniform(0, 1) > 0.5:
            pic = pic.transpose(Image.FLIP_LEFT_RIGHT)

        img = torch.ByteTensor(numpy.asarray(pic))
        img = img.view(pic.size[0], pic.size[1], 3)
        # put it in CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0,2).transpose(1,2).contiguous()

        return img, torch.IntTensor((self.data.imageClass[i],))


# demo
if __name__ == "__main__":
    import time
    num_workers = 8
    dataset = ImagenetDataset('/mnt/vol/gfsai-east/ai-group/datasets/imagenet/trainCache.t7', True)
    loader = data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=num_workers)

    start = time.time()
    i = 0
    for batch in loader:
        print("{}/{}, time= {:.04f} s".format(i, len(dataset), time.time() - start))
        i += batch[0].size(0)
        start = time.time()

    print("done")
