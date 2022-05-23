import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from torch.distributed._shard import shard_module
from torch.distributed._shard.sharded_optim import (
    ShardedOptimizer,
    named_params_with_sharded_tensor,
)
from torch.distributed._shard.sharding_plan import ShardingPlan
from torch.distributed._shard.sharding_spec import ChunkShardingSpec

"""
This is the script to test Tensor Parallel(TP) on a toy model in a
Megetron-LM SPMD style. We show an E2E working flow from forward,
backward and optimization.

More context about API designs can be found in the design:

https://github.com/pytorch/pytorch/issues/72138.

We use the example of two nn layers with an element-wise RELU in between
to show an example of Megatron-LM, which was proposed in paper:

https://arxiv.org/abs/1909.08053.

The basic idea is that we shard the first nn by column and also shard
the second nn by row so that we don't need the all gather of the result
of first nn and all scatter of input of the second nn. We can speed up
the model training by avoiding communications between two layers.

To shard a nn module, we need to create a sharding spec and plan first,
and then we shard the module based on it. We will use PyTorch native APIs
for all of them and this example shows how to use them.

Additionally, we have built an optimizer for sharded module. We show how
to use it in the example, too.
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


def _generate_sharding_spec(world_size):
    """
    We first need to create a sharding spec for our sharding work.

    For now, we only support sharding on one dimension. So we use
    ``ChunkShardingSpec`` to chunk the size of the given sharding
    dim to equally split length. The behavior is similar to
    `torch.chunk`.

    We also need to create the output sharding spec for the second nn
    because we need to aggregate(reduce) the partial result after the
    second nn layer. So we have a new sharding spec to represent that
    how we store the aggregation result in a new sharded tensor.
    """
    placements = [f"rank:{idx}/cuda:{idx}" for idx in range(world_size)]
    # Shard the first nn module's weight by dim 0.
    # (nn.Linear transposes the weight internally so dim 0 actually means column)
    colwise_spec = ChunkShardingSpec(
        dim=0,
        placements=placements,
    )
    # Shard the second nn module's weight by dim 1.
    rowwise_spec = ChunkShardingSpec(
        dim=1,
        placements=placements,
    )
    # The result from the second nn.linear layer needs aggregation by dim 0.
    output_spec = ChunkShardingSpec(
        dim=0,
        placements=placements,
    )
    return colwise_spec, rowwise_spec, output_spec


def _get_toy_module_optim(module, lr):
    """
    Creata a optimizer for sharded tensor by using ShardedOptimizer.
    """
    return ShardedOptimizer(
        dict(named_params_with_sharded_tensor(module)),
        torch.optim.SGD, # SGD is only demo purpose, one can use other optims.
        lr=lr,
    )


def _get_toy_module_sharding_plan(world_size):
    """
    The idea behind Megatron-LM is that:
    1. We shard the weight of the first nn by dim 0 (col-wise)
    2. We shard the weight of the second nn by dim 1 (row-wise)
    3. We aggregate the partial result of the second nn layer and
       store it as a sharded tensor by dim 0.
    4. Return the final result on the local shard.

    We then need to create a sharding spec based on it and
    compose a sharding plan on the basis of the spec.
    """
    colwise_spec, rowwise_spec, output_spec = _generate_sharding_spec(world_size)
    return ShardingPlan(
        # Specify the sharding plan for the component of each module.
        plan={
            "net1.weight": colwise_spec,
            "net2.weight": rowwise_spec,
        },
        # Specify the sharding plan for the output of one particular module.
        # e.g., the output of the second nn layer in the example of Megatron-LM.
        output_plan={
            "net2": output_spec,
        },
        # Specify to get the tensor stored on the local shard if the output
        # is a sharded tensor.
        return_local_tensor=["net2"],
    )


def demo_tp(rank, args):
    """
    Main body of the demo of a basic version of tensor parallel by using
    PyTorch native sharded tensor APIs.
    """
    print(f"Running basic Megatron style TP example on rank {rank}.")
    setup(rank, args.world_size)
    # create a sharding plan based on the given world_size.
    module_sharding_plan = _get_toy_module_sharding_plan(
        args.world_size
    )

    # create model and move it to GPU with id rank
    model = ToyModel().cuda(rank)
    # Shard the module based on created plan.
    shard_module(model, module_sharding_plan)
    # Create a optimizer for the sharded module.
    optimizer = _get_toy_module_optim(model, 0.002)

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
    else:
        run_demo(demo_tp, args)

