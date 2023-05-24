# Linear regression example

Trains a single fully-connected layer to fit a 4th degree polynomial.

To build the code, run the following commands from your terminal:

```shell
$ cd regression
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

where `/path/to/libtorch` should be the path to the unzipped _LibTorch_
distribution, which you can get from the [PyTorch
homepage](https://pytorch.org/get-started/locally/).

Execute the compiled binary to run:

```shell
$ ./regression
Loss: 0.000301158 after 584 batches
==> Learned function:	y = 11.6441 x^4 -3.10164 x^3 2.19786 x^2 -3.83606 x^1 + 4.37066
==> Actual function:	y = 11.669 x^4 -3.16023 x^3 2.19182 x^2 -3.81505 x^1 + 4.38219
...
```
