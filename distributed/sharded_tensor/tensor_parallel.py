import os
import tempfile
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
This is script to test Tensor Parallel(TP) on a toy model in a 
Megetron-LM SPMD style. We show a E2E working flow from forward,
backward and optimization.
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

	
def _generate_chunk_sharding_spec(world_size):
    placements = [f"rank:{idx}/cuda:{idx}" for idx in range(world_size)]
    colwise_spec = ChunkShardingSpec(
        dim=0,
        placements=placements,
    )
    rowwise_spec = ChunkShardingSpec(
        dim=1,
        placements=placements,
    )
    return colwise_spec, rowwise_spec


def _get_toy_module_optim(module, lr):
    return ShardedOptimizer(
        dict(named_params_with_sharded_tensor(module)),
        torch.optim.SGD,
        lr=lr,
    )


def _get_toy_module_sharding_plan(specs):
    colwise_spec, rowwise_spec = specs[0], specs[1]
    return ShardingPlan(
        plan={
            "net1.weight": colwise_spec,
            "net2.weight": rowwise_spec,
        },
        output_plan={
            "net2": colwise_spec,
        },
        return_local_tensor=["net2"],
    )


def demo_tp(rank, world_size):
    print(f"Running basic Megatron style TP example on rank {rank}.")
    setup(rank, world_size)
    sharding_specs = _generate_chunk_sharding_spec(world_size)
    module_sharding_plan = _get_toy_module_sharding_plan(sharding_specs)

    # create model and move it to GPU with id rank
    model = ToyModel().cuda(rank)
    shard_module(model, module_sharding_plan)
    inp = torch.rand(20, 10).cuda(rank)
    
    output = model(inp)
    output.sum().backward()
    optimizer = _get_toy_module_optim(model, 0.002)
    optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    if n_gpus < 8:
        print(f"Requires at least 8 GPUs to run, but got {n_gpus}.")
    else:
        run_demo(demo_tp, 8)

