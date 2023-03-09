# Basic Forward-Forward Example

This example implements the paper [The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/abs/2212.13345) by Geoffrey Hinton.

the aim of this paper is to introduce a new learning procedure for neural networks. the forward and backward passes of backpropagation by two forward passes.

```bash
pip install -r requirements.txt
python main.py
```

The main.py script accepts the following arguments:

```bash
optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs to train (default: 1000)
  --lr LR               learning rate (default: 0.03)
  --no_cuda             disables CUDA training
  --no_mps              disables MPS training
  --seed SEED           random seed (default: 1)
  --save_model          For saving the current Model
  --train_size TRAIN_SIZE
                        size of training set
  --threshold THRESHOLD
                        threshold for training
  --test_size TEST_SIZE
                        size of test set
  --save-model          For Saving the current Model
  --log-interval LOG_INTERVAL
                        logging training status interval
```
