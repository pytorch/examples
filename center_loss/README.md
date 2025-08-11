# Face Recognition Model with Center Loss

This example implements the center loss mentioned in the paper [A Discriminative Feature Learning Approach
for Deep Face Recognition](http://ydwen.github.io/papers/WenECCV16.pdf)

## Downloading the dataset
You can download the LFW dataset from [here](http://vis-www.cs.umass.edu/lfw/lfw.tgz) and running
```
mkdir datasets
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
cd ./datasets
mkdir -p raw
tar xvf ../lfw.tgz -C raw --strip-components=1
cd ..
```
## Runing the experiment

```
python main.py
```

## Usage
```
usage: main.py [-h] [--batch_size N] [--test_batch_size N] [--epochs N]
               [--seed S] [--max_iter S] [--lr LR] [--beta1 BETA1]
               [--center_loss_weight CENTER_LOSS_WEIGHT] [--root ROOT]
               [--resume RESUME]

PyTorch face recognition Example

optional arguments:
  -h, --help            show this help message and exit
  --batch_size N        input batch size for training (default: 128)
  --test_batch_size N   input batch size for testing (default: 64)
  --epochs N            number of epochs to train (default: 10)
  --seed S              random seed (default: 1)
  --max_iter S          random seed (default: 1)
  --lr LR               learning rate, default=0.001
  --beta1 BETA1         beta1 for adam. default=0.5
  --center_loss_weight CENTER_LOSS_WEIGHT
                        weight for center loss
  --root ROOT           path to the data directory containing aligned face
                        patches. Multiple directories are separated with
                        colon.
  --resume RESUME       model path to the resume training
```
