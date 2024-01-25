# DCGAN Example with the PyTorch C++ Frontend

This folder contains an example of training a DCGAN to generate MNIST digits
with the PyTorch C++ frontend.

The entire training code is contained in `dcgan.cpp`.

You can find the commands to install argparse [here](https://github.com/pytorch/examples/blob/main/.github/workflows/main_cpp.yml#L34).

To build the code, run the following commands from your terminal:

```shell
$ cd dcgan
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

where `/path/to/libtorch` should be the path to the unzipped _LibTorch_
distribution, which you can get from the [PyTorch
homepage](https://pytorch.org/get-started/locally/).

Execute the compiled binary to train the model:

```shell
$ ./dcgan
[ 1/30][200/938] D_loss: 0.4953 | G_loss: 4.0195
-> checkpoint 1
[ 1/30][400/938] D_loss: 0.3610 | G_loss: 4.8148
-> checkpoint 2
[ 1/30][600/938] D_loss: 0.4072 | G_loss: 4.36760
-> checkpoint 3
[ 1/30][800/938] D_loss: 0.4444 | G_loss: 4.0250
-> checkpoint 4
[ 2/30][200/938] D_loss: 0.3761 | G_loss: 3.8790
-> checkpoint 5
[ 2/30][400/938] D_loss: 0.3977 | G_loss: 3.3315
-> checkpoint 6
[ 2/30][600/938] D_loss: 0.3815 | G_loss: 3.5696
-> checkpoint 7
[ 2/30][800/938] D_loss: 0.4039 | G_loss: 3.2759
-> checkpoint 8
[ 3/30][200/938] D_loss: 0.4236 | G_loss: 4.5132
-> checkpoint 9
[ 3/30][400/938] D_loss: 0.3645 | G_loss: 3.9759
-> checkpoint 10
...
```

We can also specify the `--epochs` to change the number of epochs to train as follows:

```shell
$ ./dcgan --epochs 10
```
Without specifying the `--epochs` flag, the default number of epochs to train is 30.


The training script periodically generates image samples. Use the
`display_samples.py` script situated in this folder to generate a plot image.
For example:

```shell
$ python display_samples.py -i dcgan-sample-10.pt
Saved out.png
```
