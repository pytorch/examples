# Custom Dataset Example with the PyTorch C++ Frontend

This folder contains an example of loading a custom image dataset with OpenCV and training a model to label images, using the PyTorch C++ frontend.

The dataset used here is [Caltech 101](https://data.caltech.edu/records/mzrjq-6wc02) dataset.

The entire training code is contained in custom-data.cpp.

You can find instructions on how to install OpenCV [here](../tools/InstallingOpenCV.md).

To build the code, run the following commands from your terminal:

```shell
$ cd custom-dataset
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

where /path/to/libtorch should be the path to the unzipped LibTorch distribution, which you can get from the [PyTorch homepage](https://pytorch.org/get-started/locally/).

If you see an error like `undefined reference to cv::imread(std::string const&, int)` when running the `make` command, you should build LibTorch from source using the instructions [here](https://github.com/pytorch/pytorch#from-source), and then set `CMAKE_PREFIX_PATH` to that PyTorch source directory. An alternative solution is to use `libtorch-cxx11-abi-shared-with-deps` instead of `libtorch-shared-with-deps` as the latter is not compatible with openCV (reported [here](https://discuss.pytorch.org/t/library-conflict-between-libtorch-and-opencv/64489)).

The build directory should look like this:

```
.
├── custom-dataset
├── dataset
│   ├── accordion
│   │   ├── image_0001.jpg
│   │   ├── ...
│   ├── airplanes
│   │   ├── ...
│   ├── ...
├── info.txt
└── Makefile
└── ...
```

`info.txt` file gets copied from source directory during build.

Execute the compiled binary to train the model:

```shell
./custom-dataset
Running on: CUDA
Train Epoch: 1 16/7281	Loss: 0.314655	Acc: 0
Train Epoch: 1 176/7281	Loss: 0.532111	Acc: 0.0681818
Train Epoch: 1 336/7281	Loss: 0.538482	Acc: 0.0714286
Train Epoch: 1 496/7281	Loss: 0.535302	Acc: 0.0705645
Train Epoch: 1 656/7281	Loss: 0.536113	Acc: 0.0716463
Train Epoch: 1 816/7281	Loss: 0.537626	Acc: 0.0784314
Train Epoch: 1 976/7281	Loss: 0.537055	Acc: 0.079918
...

```