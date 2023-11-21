import os
import torch

from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from utils import ToyModel

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



"""
Main body of the demo of a basic version of sequence parallel by using
PyTorch native APIs.
"""

_rank = int(os.environ["RANK"])


def rank_print(msg):
    """helper function to print only on global rank 0"""
    if _rank==0:
        print(f"{msg}")

print(f"Running basic Megatron style Sequence Parallel example on rank {_rank}.")

# create a device mesh based on the given world_size.
_device = f"cuda"
device_mesh = init_device_mesh(device_type = _device,mesh_shape = (int(os.environ["WORLD_SIZE"]),))

rank_print(f"Device Mesh created: {device_mesh=}")


# create model and move it to GPU.  Init_device_mesh has already assigned gpu ids...
model = ToyModel().to(_device)

# Custom parallelization plan for the model
sp_model = parallelize_module(module = model,
                                device_mesh = device_mesh,
                                parallelize_plan = {
                                    "net1": ColwiseParallel(input_layouts=Shard(0)),
                                    "net2": RowwiseParallel(output_layouts=Shard(0)),
                                },
)


# Create a optimizer for the parallelized module.
lr = 0.25
optimizer = torch.optim.AdamW(sp_model.parameters(), lr=lr, foreach=True)


# Perform a num of iterations of forward/backward
# and optimizations for the sharded module.
num_iters = 10
rank_print(f"Sequence Parallel training starting...")

for i in range(num_iters):
    # For SP, input can be different across all ranks.
    inp = torch.rand(20, 10,device=_device)
    output = sp_model(inp)
    output.sum().backward()
    optimizer.step()
    rank_print(f"Sequence Parallel iter {i} completed")

rank_print(f"Sequence Parallel training completed!")
