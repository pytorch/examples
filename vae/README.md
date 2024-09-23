# Basic VAE Example

This is an improved implementation of the paper [Auto-Encoding Variational Bayes](http://arxiv.org/abs/1312.6114) by Kingma and Welling.
It uses ReLUs and the adam optimizer, instead of sigmoids and adagrad. These changes make the network converge much faster.

```bash
pip install -r requirements.txt
python main.py
```

The main.py script accepts the following arguments:

```bash
optional arguments:
  --batch-size N        input batch size for training (default: 128)
  --epochs EPOCHS       number of epochs to train (default: 10)
  --no-cuda             disables CUDA training
  --no-mps              disables macOS GPU training
  --device DEVICE       backend name
  --seed SEED           random seed (default: 1)
  --log-interval N      how many batches to wait before logging training status
```
