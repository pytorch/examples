Distributed Pipeline Parallel Example

This example shows how to distribute a ResNet50 model on two RPC workers and
then implement distributed pipeline parallelism using RPC. With pipeline
parallelism, every input batch is divided into micro-batches and thse
micro-batches are feed into the model in a pipelined fashion to increase the
amortized device utilization. Note that this example only parallelizes the
forward pass which can be viewed as the distributed counterpart of the
[single machine pipeline parallel](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html#speed-up-by-pipelining-inputs)
example.

```
pip install -r requirements.txt
python main.py
```
