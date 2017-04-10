# Deep Convolution Generative Adversarial Networks

This example implements the center loss mentioned in the paper [A Discriminative Feature Learning Approach
for Deep Face Recognition](http://ydwen.github.io/papers/WenECCV16.pdf)

After every epoch, models are saved to: `netG_epoch_%d.pth` and `netD_epoch_%d.pth`

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
