# Graph Convolutional Network

This repository contains an implementation of Graph Convolutional Networks (GCN) based on the paper "Semi-Supervised Classification with Graph Convolutional Networks" by Thomas N. Kipf and Max Welling.

## Overview
This project implements the GCN model proposed in the paper for semi-supervised node classification on graph-structured data. GCN leverages graph convolutions to aggregate information from neighboring nodes and learn node representations for downstream tasks. The implementation provides a flexible and efficient GCN model for graph-based machine learning tasks.

# Requirements
- Python 3.7 or higher
- PyTorch 2.0 or higher
- Requests 2.31 or higher
- NumPy 1.24 or higher


# Installation
```bash
pip install -r requirements.txt
python main.py
```

# Dataset
The implementation includes support for the Cora dataset, a standard benchmark dataset for graph-based machine learning tasks. The Cora dataset consists of scientific publications, where nodes represent papers and edges represent citation relationships. Each paper is associated with a binary label indicating one of seven classes. The dataset is downloaded, preprocessed and ready to use.

## Model Architecture
The GCN model architecture follows the details provided in the paper. It consists of multiple graph convolutional layers with ReLU activation, followed by a final softmax layer for classification. The implementation supports customizable hyperparameters such as the number of hidden units, the number of layers, and dropout rate.

## Usage
To train and evaluate the GCN model on the Cora dataset, use the following command:
```bash
python train.py --epochs 200 --lr 0.01 --l2 5e-4 --dropout-p 0.5 --hidden-dim 16 --val-every 20 --include-bias False --no-cuda False
```

# Results
The model achieves a classification accuracy of 82.5% on the test set of the Cora dataset after 200 epochs of training. This result is comparable to the performance reported in the original paper. However, the results can vary due to the randomness of the train/val/test split.

References
Thomas N. Kipf and Max Welling. "Semi-Supervised Classification with Graph Convolutional Networks." Link to the paper

Original paper repository: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn)