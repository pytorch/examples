# PyTorch native Tensor Parallel for distributed training

This example demonstrates SPMD Megatron-LM style Tensor Parallel by using
PyTorch native Tensor Parallel APIs, which include:

1. Simple module-level Tensor Parallelism on a dummy MLP model.
2. Simple module-level Tensor Parallelism with Sequence Parallel inputs/outputs on a dummy MLP model.
3. A E2E demo of Fully Sharded Data Parallel + Tensor Parallel (with Sequence Parallel) on a example Llama2 model.

More details about the PyTorch native Tensor Parallel APIs, please see PyTorch docs:
https://pytorch.org/docs/stable/distributed.tensor.parallel.html

```
pip install -r requirements.txt
python example.py
```
