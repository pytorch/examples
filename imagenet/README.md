# ImageNet training in PyTorch

This implements training of popular model architectures, such as ResNet, AlexNet, and VGG on the ImageNet dataset.

This is a extension version with **virtaitech/orion** vGPU support, go to https://virtaitech.com for more vGPU information

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
python main.py -a resnet18 [imagenet-folder with train and val folders]
```

The default learning rate schedule starts at 0.1 and decays by a factor of 10 every 30 epochs. This is appropriate for ResNet and models with batch normalization, but too high for AlexNet and VGG. Use 0.01 as the initial learning rate for AlexNet or VGG:

```bash
python main.py -a alexnet --lr 0.01 [imagenet-folder with train and val folders]
```

## Multi-processing Distributed Data Parallel Training

You should always use the NCCL backend for multi-processing distributed training since it currently provides the best distributed training performance.

### Single node, multiple GPUs:

```bash
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 [imagenet-folder with train and val folders]
```

### Multiple nodes:

Node 0:
```bash
python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0 [imagenet-folder with train and val folders]
```

Node 1:
```bash
python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1 [imagenet-folder with train and val folders]
```

### Single node process, multiple node GPUs
Take two node -- A & B -- with 8 GPUs each, 16 in total, as an example; original usage require us to start multiple processes on each node (A & B) with GPUs, like previous section `Multiple nodes` discribed. 

With the help of `orion` we no longer need to execute the code on both device with the same code, same data nor same development-environment. Now we only need to set up one copy of code/data/dev-env, no mather A or B node as you like.

After install the **virtaitech/orion** packages, the only thing you need to consider is the number of gpus you want to use  
by running the following command to use up all 16 gpus, without notice the distribution of gpus on several nodes.

Simply use `--virtai-crossnode` to enable the modified code, and add `--num_gpus NUM_GPUS` equals to the amount of vGPU you want to use. 
```
export ORION_VGPU=16
export ORION_RATIO=100
export ORION_GMEM=30000
export ORION_CROSS_NODE=1
time python main.py -a resnet50 --dist-url 'tcp://10.10.10.23:12345' --dist-backend 'nccl' --world-size 1 --rank 0 --epoch 1 --seed 1 --multiprocessing-distributed --virtai-crossnode  --num-gpu $ORION_VGPU /root/ImageNet_ILSVRC2012
```
This also support multiple nodes at any number, as many as you have.


## Usage

```
usage: main.py [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
               [--resume PATH] [-e] [--pretrained] [--world-size WORLD_SIZE]
               [--rank RANK] [--dist-url DIST_URL]
               [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU]
               [--multiprocessing-distributed]
               DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH, -a ARCH  model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 |
                        resnet101 | resnet152 | resnet18 | resnet34 |
                        resnet50 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                        | vgg19_bn (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training
  --virtai-crossnode
                        enable modifications from virtaitech, which 
                        Use virtai modified code to support crossnode function call,
                        which disable torch.cuda.device_count(),
                        and replace with manual-set `--num_gpus`, to run on n gpus.
                        this will also export a fake openmpi env setting `OMPI_COMM_WORLD_LOCAL_RANK`.
  --num-gpus
                        along with `--virtai_modification`,
                        to manual set number of gpus to use,
                        default with num_gpus=1
```
