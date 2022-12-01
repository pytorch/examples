# PyTorch Tensor Parallelism for distributed training

This example demonstrates SPMD Megatron-LM style tensor parallel by using
PyTorch native Tensor Parallelism APIs, which include:

1. High-level APIs for module-level parallelism with a dummy MLP model.
2. Model agnostic ops for `DistributedTensor`, such as `Linear` and `RELU`.
3. A E2E demo of tensor parallel for a given toy model (Forward/backward + optimization).

More details about the design can be found:
https://github.com/pytorch/pytorch/issues/89884

```
pip install -r requirements.txt
python example.py
```
