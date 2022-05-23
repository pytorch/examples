# PyTorch Sharder for distributed training, Tensor Parallel Example

This example demonstrates SPMD Megatron-LM style tensor parallel by using
PyTorch native sharding APIs, which include:

1. Sharding spec/plan and high-level APIs for module-level sharding.
2. Model agnostic ops for `ShardedTensor`, such as `Linear` and `RELU`.
3. A E2E demo of tensor parallel for a given toy model (Forward/backward + optimization).
4. API to optimize parameters when they are `ShardedTensor`s.


More details about the design can be found:
https://github.com/pytorch/pytorch/issues/72138


```
pip install -r requirements.txt
python main.py
```

