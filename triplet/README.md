# TFeat shallow convolutional patch descriptor

This example implements the paper [Learning local feature descriptors with
triplets and shallow convolutional neural
networks](http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf)

After every epoch, the model is saved to: `LOG_DIR/checkpoint_%d.pth`

## Requirements

You must install OpenCV with Python support

`apt-get install python-opencv`

**or**

from source
http://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html


## Usage
```
usage: main.py [-h] [--dataroot DATAROOT] [--log-dir LOG_DIR]
               [--imageSize IMAGESIZE] [--resume PATH] [--start-epoch N]
               [--epochs E] [--batch-size BS] [--test-batch-size BST] [--anchorswap]
               [--n-triplets N] [--margin MARGIN] [--lr LR] [--lr-decay LRD]
               [--wd W] [--optimizer OPT] [--no-cuda] [--gpu-id GPU_ID]
               [--seed S] [--log-interval LI]

PyTorch TFeat Example

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   path to dataset
  --log-dir LOG_DIR     folder to output model checkpoints
  --imageSize IMAGESIZE
                        the height / width of the input image to network
  --resume PATH         path to latest checkpoint (default: none)
  --start-epoch N       manual epoch number (useful on restarts)
  --epochs E            number of epochs to train (default: 10)
  --batch-size BS       input batch size for training (default: 128)
  --test-batch-size BST
                        input batch size for testing (default: 1000)
  --anchorswap          turns on anchor swap mode for triplet margin loss
  --n-triplets N        how many triplets will generate from the dataset
  --margin MARGIN       the margin value for the triplet loss function
                        (default: 2.0
  --lr LR               learning rate (default: 0.1)
  --lr-decay LRD        learning rate decay ratio (default: 1e-6
  --wd W                weight decay (default: 1e-4)
  --optimizer OPT       The optimizer to use (default: SGD)
  --no-cuda             enables CUDA training
  --gpu-id GPU_ID       id(s) for CUDA_VISIBLE_DEVICES
  --seed S              random seed (default: 0)
  --log-interval LI     how many batches to wait before logging training
                        status

```
