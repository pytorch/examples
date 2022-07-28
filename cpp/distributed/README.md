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

where /path/to/libtorch should be the path to the unzipped LibTorch distribution. Note that the LibTorch from the [PyTorch homepage] ((https://pytorch.org/get-started/locally/) does not include MPI headers and cannot be used for this example. You have to compile LibTorch manually - a set of guidelines is provided [here] (https://gist.github.com/lasagnaphil/3e0099816837318e8e8bcab7edcfd5d9), however this may vary for different systems.

To run the code,

```shell
mpirun -np {NUM-PROCS} ./dist-mnist
```
