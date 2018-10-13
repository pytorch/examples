# Character Level RNN

This project contains a character Level RNN (RNN, LSTM and GRU) language model implemented in PyTorch. It can be used to generate novel texts one character at a time. This is a PyTorch example that generates new names from scratch and can be a useful resource for learning how to easily handle sequential data with the framework.

## Getting Started

```
$ python train.py --help
Usage: train.py [OPTIONS]

  Trains a character-level Recurrent Neural Network in PyTorch.

  Args: optional arguments [python train.py --help]

Options:
  -f, --filename PATH          path for the training data file [data/names]
  -rt, --rnn-type TEXT         type of RNN layer to use [LSTM]
  -nl, --num-layers INTEGER    number of layers in RNN [2]
  -dr, --dropout FLOAT         dropout value for RNN layers [0.5]
  -es, --emb-size INTEGER      size of the each embedding [64]
  -hs, --hidden-size INTEGER   number of hidden RNN units [256]
  -n, --num-epochs INTEGER     number of epochs for training [50]
  -bz, --batch-size INTEGER    number of samples per mini-batch [32]
  -lr, --learning-rate FLOAT   learning rate for the adam optimizer [0.0002]
  -ns, --num-samples INTEGER   number of samples to generate after epoch interval [5]
  -sp, --seed-phrase TEXT      seed phrase to feed the RNN for sampling [SOS_TOKEN]
  -sa, --sample-every INTEGER  epoch interval for sampling new sequences [5]
  --help                       Show this message and exit.
```

## References

* [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [Generating Text with Recurrent Neural Networks](http://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf)