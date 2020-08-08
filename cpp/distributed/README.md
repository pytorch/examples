# Distributed Training on MNIST using PyTorch C++ Frontend (Libtorch)

This folder contains an example of data-parallel training of a convolutional neural network on the MNIST dataset. For parallelization, Message Passing Interface (MPI) is used.

The entire code is contained in dist-mnist.cpp

You can find instructions on how to install MPI [here] (https://www.open-mpi.org/faq/?category=building). This code was tested on Open MPI but it should run on other MPI distributions as well such as MPICH, MVAPICH, etc.

To build the code, run the following commands from the terminal:

```shell
$ cd distributed
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

where /path/to/libtorch should be the path to the unzipped LibTorch distribution, which you can get from the [PyTorch homepage] ((https://pytorch.org/get-started/locally/).

To run the code,

```shell
mpirun -np {NUM-PROCS} ./dist-mnist
```

