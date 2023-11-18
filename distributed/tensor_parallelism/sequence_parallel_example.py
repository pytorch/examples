import argparse
import os
import torch

from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard

from torch.distributed.tensor.parallel import (
    PairwiseParallel,
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from utils import cleanup, ToyModel, torchrun_setup

try:
    from torch.distributed.tensor.parallel import (
        SequenceParallel
    )
    SP_AVAILABLE = True
except BaseException as e:
    pass


"""
This is the script to test Sequence Parallel(SP) on a toy model in a
Megetron-LM SPMD style. We show an E2E working flow from forward,
backward and optimization.

We use the example of two `nn.Linear` layers with an element-wise `nn.RELU`
in between to show an example of sequence parallel, which was proposed in paper:

https://arxiv.org/pdf/2205.05198.pdf.

Like tensor parallel, we parallelize the first linear layer by column
and also parallelize the second linear layer by row. But the input in each rank
now is different so that we need one all-gather for input and one reduce-scatter
in the end of the second linear layer.
"""


def demo_sp(args):
    """
    Main body of the demo of a basic version of sequence parallel by using
    PyTorch native APIs.
    """
    torchrun_setup()

    # understand world topology
    _rank = int(os.environ["RANK"])
    _local_rank = int(os.environ["LOCAL_RANK"])
    _world_size = int(os.environ["WORLD_SIZE"])
    _local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    torch.cuda.set_device(_local_rank)

    def rank_print(msg):
        """helper function to print only on global rank 0"""
        if _rank==0:
            print(f"{msg}")

    print(f"Running basic Megatron style TP example on rank {_rank}.")

    # create a sharding plan based on the given world_size.
    device_mesh = DeviceMesh("cuda", torch.arange(0, _world_size))

    device = f"cuda"
    #device_mesh = init_device_mesh(device, torch.arange(0,_world_size)) # , mesh_dim_names=("sp",))
    assert device_mesh is not None, "unable to create valid device mesh"

    rank_print(f"Device Mesh created: {device_mesh=}")


    # create model and move it to GPU with id rank
    model = ToyModel().cuda(_rank)

    # Custom parallelization plan for the model
    sp_model = parallelize_module(module = model,
                                    device_mesh = device_mesh,
                                    parallelize_plan = {
                                        "net1": ColwiseParallel(input_layouts=Shard(0)),
                                        "net1": RowwiseParallel(input_layouts=Shard(0)),
                                    },
    )


    # Create a optimizer for the parallelized module.
    lr = 0.25
    optimizer = torch.optim.AdamW(sp_model.parameters(), lr=lr)


    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    num_iters = 10
    rank_print(f"Sequence Parallel training starting...")

    for i in range(num_iters):
        # For SP, input can be different across all ranks.
        inp = torch.rand(20, 10).cuda(_rank)
        output = sp_model(inp)
        output.sum().backward()
        optimizer.step()
        rank_print(f"Sequence Parallel iter {i} completed")

    rank_print(f"Sequence Parallel training completed!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # This is passed in via cmd
    args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    demo_sp(args,)
