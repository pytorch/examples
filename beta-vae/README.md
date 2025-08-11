# Basic beta-VAE Example

This is an implementation of the paper [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/pdf?id=Sy2fzU9gl).

We did experimentation on the FashionMNIST dataset, using a very simple convolution neural network architecture.

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
  --seed		random seed (default: 1)
  --log-interval	how many batches to wait before logging training status
```