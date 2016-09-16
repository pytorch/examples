import math
import random

import torch
import torch.multiprocessing as multiprocessing

import os.path
import torchfile
import numpy
from PIL import Image

###########################################################
# These widgets go in some dataset library
###########################################################
class Dataset(object):

    def size(self):
        raise NotImplementedError()

    def get(self, i):
        raise NotImplementedError()

class PermutedDataset(Dataset):

    def __init__(self, dataset, perm=None):
        self.dataset = dataset
        self.perm = perm or torch.randperm(dataset.size())

    def size(self):
        return self.dataset.size()

    def get(self, i):
        return self.dataset.get(int(self.perm[i]))

class PartitionedDataset(Dataset):

    def __init__(self, dataset, part, nPart):
        self.dataset = dataset
        self.start = dataset.size() * part / nPart
        self.end   = dataset.size() * (part+1) / nPart

    def size(self):
        return self.end - self.start

    def get(self, i):
        return self.dataset.get(self.start + i)

###########################################################
# This is the main imagenet loading logic
###########################################################
class ImagenetDataset(Dataset):
    
    def __init__(self, path, jitter):
        self.path = path 
        self.data = torchfile.load(path)
        self.jitter = jitter
        self.res = 256

    def size(self):
        # return 1000
        return len(self.data.imagePath)

    def get(self, i):
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
        img = img.transpose(0,2).transpose(1,2).contiguous() # put it in CHW format

        # lets wait until we have Python bindings for torch.image to do scale/crop
        return img, self.data.imageClass[i]


###########################################################
# Where does this widget go?
###########################################################
class MultiQueueIterator(object):

    def __init__(self, queue, N, sentinel=None):
        self.queue = queue
        self.N = N
        self.i = 0
        self.sentinel = sentinel
    
    def __iter__(self):
        return self

    def next(self):
        while self.i < self.N:
            e = self.queue.get()
            if e == self.sentinel:
                self.i += 1
            else:
                return e
        raise StopIteration()


###########################################################
# Shim that runs in each process
###########################################################
def _dataLoader(queue, dataset):
    batchSize = 64
    for i in range(0, dataset.size(), batchSize):
        batch = [dataset.get(x) for x in range(i, i + batchSize) if x < dataset.size()]
        queue.put(zip(*batch))
    queue.put(None)


###########################################################
# This is what's called externally
###########################################################
def makeDataIterator(datasetPath, isTest, nProc):
    dataset = PermutedDataset(ImagenetDataset(datasetPath, not isTest))
    queue = multiprocessing.Queue()
    processes = [multiprocessing.Process(target=_dataLoader, 
        args=(queue, PartitionedDataset(dataset, i, nProc))).start() for i in range(nProc)]
    return dataset, MultiQueueIterator(queue, nProc)

# demo
if __name__ == "__main__":
    import time
    nDonkeys = 8
    dataset, dataIterator = makeDataIterator(
        '/mnt/vol/gfsai-east/ai-group/datasets/imagenet/trainCache.t7',
        False, nDonkeys)

    start = time.time()
    i = 0
    for images, labels in dataIterator:
        print("{}/{}, time= {:.04f} s".format(i, dataset.size(), time.time() - start))
        i += len(images)
        start = time.time()
