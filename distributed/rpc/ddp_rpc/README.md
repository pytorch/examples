Distributed DataParallel + Distributed RPC Framework Example

The example shows how to combine Distributed DataParallel with the Distributed 
RPC Framework. There are two trainer nodes, 1 master node and 1 parameter 
server in the example.

The master node creates an embedding table on the parameter server and drives 
the training loop on the trainers. The model consists of a dense part 
(nn.Linear) replicated on the trainers via Distributed DataParallel and a 
sparse part (nn.EmbeddingBag) which resides on the parameter server. Each 
trainer performs an embedding lookup on the parameter server (using the 
Distributed RPC Framework)  and then executes its local nn.Linear module. 
During the backward pass, the gradients for the dense part are aggregated via 
allreduce by DDP and the distributed backward pass updates the parameters for 
the embedding table on the parameter server.


```
pip install -r requirements.txt
python main.py
```
