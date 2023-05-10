import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

TP_AVAILABLE = False
try:
    from torch.distributed._tensor import DeviceMesh
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.tensor.parallel import (
        PairwiseParallel,
        parallelize_module,
        SequenceParallel,
    )
    from torch.distributed.tensor.parallel.fsdp import enable_2d_with_fsdp

    TP_AVAILABLE = True
except BaseException as e:
    pass


"""
This is the script to test Tensor Parallel(TP) on a toy model in a
Megetron-LM SPMD style. We show an E2E working flow from forward,
backward and optimization.

More context about API designs can be found in the design:

https://github.com/pytorch/pytorch/issues/89884.

And it is built on top of Distributed Tensor which is proposed in:

https://github.com/pytorch/pytorch/issues/88838.

We use the example of two `nn.Linear` layers with an element-wise `nn.RELU`
in between to show an example of Megatron-LM, which was proposed in paper:

https://arxiv.org/abs/1909.08053.

The basic idea is that we parallelize the first linear layer by column
and also parallelize the second linear layer by row so that we only need
one all reduce in the end of the second linear layer.

We can speed up the model training by avoiding communications between
two layers.

To parallelize a nn module, we need to specify what parallel style we want
to use and our `parallelize_module` API will parse and parallelize the modules
based on the given `ParallelStyle`. We are using this PyTorch native Tensor
Parallelism APIs in this example to show users how to use them.

Also one can specify the flag `run_seq_parallel` to run the example about
sequence parallel(SP) like what Megatron-LM Sequence parallel
(https://arxiv.org/pdf/2205.05198.pdf) is doing.

We also show an example of 2D parallel with the flag `run_2d`.
We enabled Fully Sharded Data Parallel + Tensor Parallel in
separate parallel dimensions:
    Data Parallel across hosts
    Tensor Parallel within each host

 We use a simple diagram to illustrate below:

======================================================================
------------       ------------       ------------       ------------
| Host 1   |       | Host 2   |       |          |       | Host N   |
| 8 GPUs   |       | 8 GPUs   |       |          |       | 8 GPUs   |
|          |       |          |       |    ...   |       |          |
| (TP)     |       | (TP)     |       |          |       | (TP)     |
|[0,1,..,7]|       |[8,9..,15]|       |          |       |[8N-8,8N-7|
|          |       |          |       |          |       | .., 8N-1]|
|          |       |          |       |          |       |          |
------------       ------------       ------------       ------------
FSDP:
[0, 8, ..., 8N-8], [1, 9, ..., 8N-7], ..., [7, 15, ..., 8N-1]
======================================================================

More details can be seen in the slide:
https://docs.google.com/presentation/d/17g6WqrO00rP3MsxbRENsPpjrlSkwiA_QB4r93_eB5is/
"""


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(32, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def generate_mesh(world_size, is_2d):
    mesh = torch.arange(0, world_size)
    if not is_2d:
        return mesh
    else:
        assert world_size % 2 == 0, "Need to have 2N GPU to enable 2D demo."
        # TP_size = 2.
        return mesh.view(-1, 2)


def demo(rank, args):
    """
    Main body of the demo of a basic version of tensor parallel by using
    PyTorch native APIs.
    """
    print(f"Running basic Megatron style TP example on rank {rank}.")
    setup(rank, args.world_size)

    # create a sharding plan based on the given world_size.
    device_mesh = DeviceMesh("cuda", generate_mesh(args.world_size, args.run_2d))

    # create model and move it to GPU with id rank
    model = ToyModel().cuda(rank)
    # Create a optimizer for the parallelized module.
    LR = 0.25
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    # Parallelize the module based on the given Parallel Style.
    parallel_style = SequenceParallel() if args.run_seq_parallel else PairwiseParallel()
    model = parallelize_module(model, device_mesh, parallel_style)

    if args.run_2d:
        # We need to register hooks for TP + FSDP integration.
        assert (
            enable_2d_with_fsdp()
        ), "FSDP 2D hook is not registered. Please use PyTorch with version >= 2.0"
        model = FSDP(model)

    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    for _ in range(args.iter_nums):
        # For TP, input needs to be same across all TP ranks.
        # while for SP, input can be different across all ranks.
        inp = torch.rand(20, 10).cuda(rank)
        output = model(inp)
        output.sum().backward()
        optimizer.step()

    cleanup()


def run_demo(demo_fn, args):
    mp.spawn(demo_fn, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    # This is passed in via cmd
    parser.add_argument("--world_size", type=int, default=n_gpus)
    parser.add_argument("--iter_nums", type=int, default=10)
    parser.add_argument("--run_seq_parallel", type=bool, default=False)
    parser.add_argument("--run_2d", type=bool, default=False)
    args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    if n_gpus < 2:
        print("Requires at least 2 GPUs to run.")
    elif not TP_AVAILABLE:
        print(
            "PyTorch doesn't have Tensor Parallelism available," " need nightly build."
        )
    else:
        run_demo(demo, args)
