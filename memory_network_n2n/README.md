# End to End Memory Network
This repository contains a pytorch implementation of an End to End Memory Network.

The model uses the method described in [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895).

The tasks are from the [bAbl](http://arxiv.org/abs/1502.05698) dataset.

![MemN2N picture](https://www.dropbox.com/s/s9txttvrdkhmtkt/memn2n.png?dl=0)

## Requirements
The program is written in Python, and uses [pytorch](http://pytorch.org/), [scikit-learn](https://scikit-learn.org/). A GPU is not necessary, but can provide a significant speed up especially during training.

## Usage
Train and Evaluate model
```
python main.py --train 1 --lr 0.001 --hops 3 --eval 1 --saved-model-dir ./saved/ --data-dir ./data/tasks_1-20_v1-2/en-10k --task-number 1
```
* `--epochs`: number of epochs to train for, default : 100
* `--train`: to train or not, default : 1
* `--lr`: set the learning rate, default : 0.001
* `--hops`: number of hops in the memory network, default : 1
* `--eval`: evaluate against testing data, default : 1
* `--saved-model-dir`: directory to save the model to, default : ./saved/
* `--data-dir`: data directory which holds the tasks, default : ./data/tasks_1-20_v1-2/en-10k
* `--task-number`: task on which to train, default : 1


There are several command line arguments, the important ones are listed below
* `--joint-training`: enable joint training, default: 0
* `--batch-size`: batch-size for training, default: 32
* `--embed-size`: embedding dimensions, default: 25
* `--anneal-factor`: factor to anneal by every 'anneal-epoch(s)', default: 2
* `--anneal-epoch`: anneal every `anneal-epoch` epoch, default: 25
* `--log-epochs`: Number of epochs after which to log progress, default: 4
* `--debug`: Set to 1 for debugging purposes - print weight and other matrices, default : 0
* `--saved-model-dir`: path to folder where trained model will be saved.
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.

Refer to ``memory_network_n2n/main.py`` for other command line arguments.

## Results

Task  |  Testing Accuracy
------|------------------
1     |  99.39
2     |  27.92
3     |  24.69
4     |  95.76
5     |  80.44
6     |  91.13
7     |  82.66
8     |  82.15
9     |  88.5
10    |  41.43
11    |  90.72
12    |  99.69
13    |  94.25
14    |  97.27
15    |  100
16    |  47.98
17    |  57.15
18    |  73.68
19    |  11.39
20    |  80.84
