# Graph Attention Network

This repository contains a PyTorch implementation of the **Graph Attention Networks (GAT)** based on the paper ["Graph Attention Network" by Velickovic et al](https://arxiv.org/abs/1710.10903v3). 

The Graph Attention Network is a powerful graph neural network model for learning represtations on graph-structured data, which has shown excellent performance in various tasks such as node classification, link prediction, and graph classification.


## Overview
The Graph Attention Network (GAT) is a graph neural network architecture designed specifically for handling graph-structured data. It leverages multi-head attention mechanism to capture the information of neighboring nodes in an attentive manner to learn represtations for each node. This attention mechanism allows the model to focus on relevant nodes and adaptively weight their contributions during message passing.

Check out the following resources for more ino on GATs:
- [Blog post by the main auther, Petar Velickovic](https://petar-v.com/GAT/)
- [Main paper](https://doi.org/10.48550/arXiv.1710.10903)

This repository provides a clean and short implementation of the official GAT model using PyTorch. The code is well-documented and easy to understand, making it a valuable resource for researchers and practitioners interested in graph deep learning.


## Key Features

- **GAT Model**: Implementation of the Graph Attention Network model with multi-head attention based on the paper "Graph Attention Network" by Velickovic et al.
- **Graph Attention Layers**: Implementation of graph convolutional layers that aggregate information from neighboring nodes using a self-attention mechanisms to learn node importance weights.
- **Training and Evaluation**: Code for training GAT models on graph-structured data and evaluating their performance on node classification tasks on the *Cora* benchmark dataset.

---

# Requirements
- Python 3.7 or higher
- PyTorch 2.0 or higher
- Requests 2.31 or higher
- NumPy 1.24 or higher



# Dataset
The implementation includes support for the Cora dataset, a standard benchmark dataset for graph-based machine learning tasks. The Cora dataset consists of scientific publications, where nodes represent papers and edges represent citation relationships. Each paper is associated with a binary label indicating one of seven classes. The dataset is downloaded, preprocessed and ready to use.

# Model Architecture
The official architecture (used in this project) proposed in the paper "Graph Attention Network" by Velickovic et al. consists of two graph attention layers which incorporates the multi-head attention mechanisms during its message trasformation and aggregation. Each graph attention layer applies a shared self-attention mechanism to every node in the graph, allowing them to learn different representations based on the importance of their neighbors.

In terms of activation functions, the GAT model employs both the **Exponential Linear Unit (ELU)** and the **Leaky Rectified Linear Unit (LeakyReLU)** activations, which introduce non-linearity to the model. ELU is used as the activation function for the **hidden layers**, while LeakyReLU is applied to the **attention coefficients** to ensure non-zero gradients for negative values.

Following the official implementation, the first GAT layer consists of **K = 8 attention heads** computing **F' = 8 features** each (for a **total of 64 features**) followed by an exponential linear unit (ELU) activation on the layer outputs. The second GAT layer is used for classification: a **single attention head** that computes C features (where C is the number of classes), followed by a softmax activation for probablisitic outputs. (we use log-softmax instead for computational convenience with using NLLLoss)

*Note that due to being an educational example, this implementation uses the full dense form of the adjacency matrix of the graph, and not the sparse form of the matrix. Thus all the operations in the model implemeation is done in a non-sparse from. This will not affect the model's performance accuracy-wise. However an sparse-friendly implementation will help with the efficiency in the use of resources, storage, and speed.*
 

# Usage
Training and evaluating the GAT model on the Cora dataset can be done through running the `main.py` script as follows:

1. Clone the PyTorch examples repository:

```
git clone https://github.com/pytorch/examples.git
cd examples/gat
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Train the GAT model by running the `main.py` script as follows:: (Example using the default parameters)

```bash
python main.py --epochs 300 --lr 0.005 --l2 5e-4 --dropout-p 0.6 --num-heads 8 --hidden-dim 64 --val-every 20
```

In more detail, the `main.py` script recieves following arguments:
```
usage: main.py [-h] [--epochs EPOCHS] [--lr LR] [--l2 L2] [--dropout-p DROPOUT_P] [--hidden-dim HIDDEN_DIM] [--num-heads NUM_HEADS] [--concat-heads] [--val-every VAL_EVERY]
               [--no-cuda] [--no-mps] [--dry-run] [--seed S]

PyTorch Graph Attention Network

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs to train (default: 300)
  --lr LR               learning rate (default: 0.005)
  --l2 L2               weight decay (default: 6e-4)
  --dropout-p DROPOUT_P
                        dropout probability (default: 0.6)
  --hidden-dim HIDDEN_DIM
                        dimension of the hidden representation (default: 64)
  --num-heads NUM_HEADS
                        number of the attention heads (default: 4)
  --concat-heads        wether to concatinate attention heads, or average over them (default: False)
  --val-every VAL_EVERY
                        epochs to wait for print training and validation evaluation (default: 20)
  --no-cuda             disables CUDA training
  --no-mps              disables macOS GPU training
  --dry-run             quickly check a single pass
  --seed S              random seed (default: 13)
```



# Results
After training for **300 epochs** with default hyperparameters on random train/val/test data splits, the GAT model achieves around **%81.25** classification accuracy on the test split. This result is comparable to the performance reported in the original paper. However, the results can vary due to the randomness of the train/val/test split.

# Reference

``` 
@article{
  velickovic2018graph,
  title="{Graph Attention Networks}",
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`{o}}, Pietro and Bengio, Yoshua},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJXMpikCZ},
}
```
- Paper on arxiv: [arXiv:1710.10903v3](https://doi.org/10.48550/arXiv.1710.10903)
- Original paper repository: [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT)
