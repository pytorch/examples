# PyTorch native Tensor Parallel for distributed training

This example demonstrates SPMD Megatron-LM style Tensor Parallel by using
PyTorch native Tensor Parallel APIs, which include:

1. Simple module-level Tensor Parallelism on a dummy MLP model.
2. Simple module-level Tensor Parallelism with Sequence Parallel inputs/outputs on a dummy MLP model.
3. A E2E demo of Fully Sharded Data Parallel + Tensor Parallel (with Sequence Parallel) on a example Llama2 model.

More details about the PyTorch native Tensor Parallel APIs, please see PyTorch docs:
https://pytorch.org/docs/stable/distributed.tensor.parallel.html

## Installation

```bash
pip install -r requirements.txt
```

## Running Examples

You can run the examples using `torchrun` to launch distributed training:

```bash
# Simple Tensor Parallel example
torchrun --nnodes=1 --nproc_per_node=4 tensor_parallel_example.py

# Tensor Parallel with Sequence Parallel
torchrun --nnodes=1 --nproc_per_node=4 sequence_parallel_example.py

# FSDP + Tensor Parallel with Llama2 model
torchrun --nnodes=1 --nproc_per_node=4 fsdp_tp_example.py
```

For more details, check the `run_examples.sh` script.
