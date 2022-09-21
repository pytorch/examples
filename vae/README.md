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
  --batch-size		input batch size for training (default: 128)
  --epochs		number of epochs to train (default: 10)
  --no-cuda		enables CUDA training
  --mps         enables GPU on macOS
  --seed		random seed (default: 1)
  --log-interval	how many batches to wait before logging training status
```