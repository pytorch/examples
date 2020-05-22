# Sentence Classification tutorial with the PyTorch c++ Frontend
This folder contains an example of training a Convolutional Neural Network for sentence classification on TREC dataset, using the PyTorch C++ frontend.

The code is split into 2 folders:

```src``` folder contains the files ```datautils.cpp``` and ```sentence_classification.cpp```

```include``` folder contains the files ```datautils.h```, ```loader.h```, ```model.h``` and ```trainmodel.h```

To build the code, run the following commands from your terminal:
```
$ cd sentence_classification
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```
where ```/path/to/libtorch``` should be the path to the unzipped LibTorch distribution, which you can get from the [PyTorch homepage](https://pytorch.org/get-started/locally/).

Execute the compiled binary to train the model:
```
./sentence_classification
Loaded Training data
Loaded Testing data
Training Data Size: 5452
Test Data Size: 500
No GPU available. Training on CPU. 
Epoch: 1 Training Loss: 1.001
Epoch: 2 Training Loss: 0.263
Epoch: 3 Training Loss: 0.097
Epoch: 4 Training Loss: 0.047
Epoch: 5 Training Loss: 0.022
Epoch: 6 Training Loss: 0.017
Epoch: 7 Training Loss: 0.009
Epoch: 8 Training Loss: 0.006
Epoch: 9 Training Loss: 0.005
Epoch: 10 Training Loss: 0.004
Test Loss: 0.049 Test Accuracy: 0.840
```
