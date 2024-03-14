# distributed-pytorch

Code for the DDP tutorial series at https://pytorch.org/tutorials/beginner/ddp_series_intro.html

Each code file extends upon the previous one. The series starts with a non-distributed script that runs on a single GPU and incrementally updates to end with multinode training on a Slurm cluster.

## Dependencies

1. nccl
   1. https://github.com/NVIDIA/nccl
   2. https://github.com/NVIDIA/nccl-tests
2. torch>=1.11.0


## Files
* [single_gpu.py](single_gpu.py): Non-distributed training script

* [multigpu.py](multigpu.py): DDP on a single node

* [multigpu_torchrun.py](multigpu_torchrun.py): DDP on a single node using Torchrun

* [multinode.py](multinode.py): DDP on multiple nodes using Torchrun (and optionally Slurm)
    * [slurm/setup_pcluster_slurm.md](slurm/setup_pcluster_slurm.md): instructions to set up an AWS cluster
    * [slurm/config.yaml.template](slurm/config.yaml.template): configuration to set up an AWS cluster
    * [slurm/sbatch_run.sh](slurm/sbatch_run.sh): slurm script to launch the training job

## Create Virtual Environment

```shell
$ python -m venv </path/to/new/virtual/environment>
$ source </path/to/new/virtual/environment>/bin/activate
```

## Run commands

* [single_gpu.py](single_gpu.py):
```shell
$ python single_gpu.py 50 10
```

* [multigpu.py](multigpu.py):

```shell
$ python multigpu.py 50 10
```


* [multigpu_torchrun.py](multigpu_torchrun.py):
```shell
$ torchrun --standalone --nproc_per_node=gpu multigpu_torchrun.py 50 10
```

* [multinode.py](multinode.py): DDP on multiple nodes using Torchrun (and optionally Slurm)

  TODO
