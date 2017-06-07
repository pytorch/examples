Residual Networks
=================

The code implements Residual Networks https://arxiv.org/abs/1603.05027
and Wide Residual Networks http://arxiv.org/abs/1605.07146 training on CIFAR.

The architecture used is pre-activation basic-block ResNet, and accuracy
is the same with lua version of WRN training code
<https://github.com/szagoruyko/wide-residual-networks>

The code doesn't use `torch.nn.modules` and works by initializing
all parameter tensors and defining a function that utilizes these
tensors with `torch.nn.functional` interface.


## Requirements

Install PyTorch and PyTorchNet: https://github.com/pytorch/tnt

Then do:

```
pip install -r requirements.txt
```


## Details

#### Preprocessing

The code does mean-std normalization as 
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch)
and applies random flips with random crops from an image padded with 4 pixels on
each side. Padding is done with reflections.

#### Training example

To train WRN with 16 layers and width 4 on 2 GPUs do:

```
python main.py --save logs/resnet_16_4 --depth 16 --width 4 --gpu_id 0,1 --ngpu 2
``` 

#### Reference results

We rerun the experiments from the WRN paper with PyTorch (5-time run results, no dropout):

| Model | CIFAR-10 | CIFAR-100 |
|:-----:|:--------:|:---------:|
|WRN-40-4 | 4.53 | 21.18 |
|WRN-16-8 | 4.27 | 20.43 |
|WRN-28-10 | 4.00 | 19.25 |


## Multi-GPU and CPU support

Both are supported, for CPU you might want to try a very thin ResNet with
 `--depth 16 --width 0.5`, because it's much slower than GPU.
