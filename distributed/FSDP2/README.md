## FSDP2
To run FSDP2 on transformer model:
```
torchrun --nproc_per_node 2 train.py
```

## Ensure you are running a recent version of PyTorch:
see https://pytorch.org/get-started/locally/ to install at least 2.5 and ideally a current nightly build.
