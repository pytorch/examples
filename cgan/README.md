# Conditional Generative Adversarial Networks 

This example implements a conditional generative adversarial network, as illustrated in [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)

This implementation is very close to the [dcgan implementation](https://github.com/pytorch/examples/tree/master/dcgan).

After every 100 training iterations, the files `real_samples.png` and `fake_samples_%3d.png` are written to disk
with the samples from the generative model.

After every epoch, models are saved to: `netG_epoch_%d.pth` and `netD_epoch_%d.pth`

## Downloading the dataset
You can download the MNIST dataset [here](http://yann.lecun.com/exdb/mnist) 

You can download the LSUN dataset by cloning [this repo](https://github.com/fyu/lsun) and running
```
python download.py -c bedroom
```

## Usage
```
usage: main.py [-h] --dataset DATASET --dataroot DATAROOT 
               [--batchSize BATCHSIZE] [--imageSize IMAGESIZE] [--channels CHANNELS]
               [--latentdim LATENDIM] [--n_classes N_CLASSES] [--epoch EPOCH] [--lrte LRATE]
               [--beta1 BETA1] [--cuda] [--ngpu NGPU] [--netG NETG]
               [--netD NETD]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     cifar10 | lsun | mnist
  --dataroot DATAROOT   path to dataset
  --batchSize BATCHSIZE 
                        input batch size
  --imageSize IMAGESIZE
                        the height / width of the input image to network
  --channels CHANNELS   the channels of the input image to network 
  --latentdim LATENTDIM the size of the latent vector 
  --n_classes N_CLASSES the number of classes/labels in the dataset
  --epoch EPOCH         number of epochs to train for
  --lrate LRATE         learning rate, default=0.0002
  --beta BETA           beta for adam. default=0.5
  --beta1 BETA1         beta1 for adam. default=0.999
  --output              folder to output images. defualt=.
  --randomseed          
