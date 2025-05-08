## FSDP2
To run FSDP2 on transformer model:
```
cd distributed/FSDP2
torchrun --nproc_per_node 2 train.py
```
* For 1st time, it creates a "checkpoints" folder and save state dicts there
* For 2nd time, it loads from previous checkpoints

To enable explicit prefetching
```
torchrun --nproc_per_node 2 train.py --explicit-prefetch
```

To enable mixed precision
```
torchrun --nproc_per_node 2 train.py --mixed-precision
```

To showcse DCP API
```
torchrun --nproc_per_node 2 train.py --dcp-api
```

## Ensure you are running a recent version of PyTorch:
see https://pytorch.org/get-started/locally/ to install at least 2.5 and ideally a current nightly build.
