# Own Dataset Example with the PyTorch C++ Frontend

This folder contains an example of making an origianl image dataset to training classification model using PyTorch C++ frontend.

The entire dataset code is contained in dataset.cpp

To build the code, run the following commands from your terminal.

```bash
$ cd own_dataset
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

where /path/to/libtorch should be the path to the unzipped LibTorch distribution, which you can get from the [PyTorch homepage](https://pytorch.org/get-started/locally/).

Execute the compiled binary to test own dataset:

```bash
$ ./dataset ../test_data/ ../test_data/labels.txt
input dim: 4
target:  1
[ Variable[CPUByteType]{1} ]
```

The ```test_data``` directory has the following structure.

```bash
$ tree ../test_data/
../test_data/
|-- image1.jpg
|-- image2.jpg
|-- image3.jpg
`-- labels.txt

0 directories, 4 files
```

The contents of the ```labels.txt``` have the following format.

```bash
$ cat ../test_data/labels.txt
image1.jpg,0
image2.jpg,0
image3.jpg,1
```
