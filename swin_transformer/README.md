# Swin Transformer on CIFAR-10

This project demonstrates a minimal implementation of a **Swin Transformer** for image classification on the **CIFAR-10** dataset using PyTorch.

It includes:
- Patch embedding and window-based self-attention
- Shifted windows for hierarchical representation
- Training and testing logic using standard PyTorch utilities

---

## Files

- `swin_transformer.py` — Full implementation of the Swin Transformer model, training loop, and evaluation on CIFAR-10.
- `README.md` — This file.

---

## Requirements

- Python 3.8+
- PyTorch 2.6 or later
- `torchvision` (for CIFAR-10 dataset)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Train & Save the model

```bash
python swin_transformer.py --epochs 10 --batch-size 64 --lr 0.001 --save-model
```

### Test the model

Testing is done automatically after each epoch. To only test, run with:

```bash
python swin_transformer.py --epochs 1
``

The model will be saved as `swin_cifar10.pt`.

---

## Features

- Uses shifted window attention for local self-attention.
- Patch-based embedding with a lightweight network.
- Trains on CIFAR-10 with `Adam` optimizer and learning rate scheduling.
- Prints loss and accuracy per epoch.

---

