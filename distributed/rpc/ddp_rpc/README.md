Distributed DataParallel + Distributed RPC Framework Example

This example demonstrates how to combine Distributed DataParallel (DDP) with the Distributed RPC Framework. It requires two trainer nodes (each with a GPU), one master node, and one parameter server.

The master node initializes an embedding table on the parameter server and orchestrates the training loop across the trainers. The model is composed of a dense component (`nn.Linear`), which is replicated on the trainers using DDP, and a sparse component (`nn.EmbeddingBag`), which resides on the parameter server.

Each trainer performs embedding lookups on the parameter server via RPC, then processes the results through its local `nn.Linear` module. During the backward pass, DDP aggregates gradients for the dense part using allreduce, while the distributed backward pass updates the embedding table parameters on the parameter server.


```
pip install -r requirements.txt
python main.py
```
