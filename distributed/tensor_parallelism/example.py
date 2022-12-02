import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

TP_AVAILABLE = False
try:
    from torch.distributed._tensor import (
        DeviceMesh,
    )
    from torch.distributed.tensor.parallel import (
        PairwiseParallel,
        parallelize_module,
    )
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
"""


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

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


def demo_tp(rank, args):
    """
    Main body of the demo of a basic version of tensor parallel by using
    PyTorch native APIs.
    """
    print(f"Running basic Megatron style TP example on rank {rank}.")
    setup(rank, args.world_size)
    # create a sharding plan based on the given world_size.
    device_mesh = DeviceMesh(
        "cuda",
        torch.arange(args.world_size),
    )

    # create model and move it to GPU with id rank
    model = ToyModel().cuda(rank)
    # Create a optimizer for the parallelized module.
    LR = 0.25
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    # Parallelize the module based on the given Parallel Style.
    model = parallelize_module(model, device_mesh, PairwiseParallel())

    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    for _ in range(args.iter_nums):
        inp = torch.rand(20, 10).cuda(rank)
        output = model(inp)
        output.sum().backward()
        optimizer.step()

    cleanup()


def run_demo(demo_fn, args):
    mp.spawn(demo_fn,
             args=(args,),
             nprocs=args.world_size,
             join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    # This is passed in via cmd
    parser.add_argument("--world_size", type=int, default=n_gpus)
    parser.add_argument("--iter_nums", type=int, default=10)
    args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    if n_gpus < 2:
        print("Requires at least 2 GPUs to run.")
    elif not TP_AVAILABLE:
        print(
            "PyTorch doesn't have Tensor Parallelism available,"
            " need nightly build."
        )
    else:
        run_demo(demo_tp, args)

