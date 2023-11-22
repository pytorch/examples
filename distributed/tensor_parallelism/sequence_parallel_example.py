import os
import sys
import torch
import torch.nn as nn

from torch.distributed._tensor import Shard

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

from log_utils import rank_log, get_logger, verify_min_gpu_count


# ---- GPU check ------------
_min_gpu_count = 2

if not verify_min_gpu_count(min_gpus=_min_gpu_count):
    print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
    sys.exit()
# ---------------------------


from torch.distributed._tensor.device_mesh import init_device_mesh



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


class ToyModel(nn.Module):
    """MLP based model"""

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(32, 5)

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))


"""
Main body of the demo of a basic version of sequence parallel by using
PyTorch native APIs.
"""
logger = get_logger()

# create a device mesh based on the given world_size.
device_mesh = init_device_mesh(
    device_type="cuda", mesh_shape=(int(os.environ["WORLD_SIZE"]),)
)

_rank = device_mesh.get_rank()

print(f"Starting PyTorch Sequence Parallel example on rank {_rank}.")

rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")

# create model and move it to GPU.  Init_device_mesh has already assigned gpu ids...
model = ToyModel().to("cuda")

# Custom parallelization plan for the model
sp_model = parallelize_module(
    module=model,
    device_mesh=device_mesh,
    parallelize_plan={
        "in_proj": ColwiseParallel(input_layouts=Shard(0)),
        "out_proj": RowwiseParallel(output_layouts=Shard(0)),
    },
)


# Create a optimizer for the parallelized module.
lr = 0.25
optimizer = torch.optim.AdamW(sp_model.parameters(), lr=lr, foreach=True)


# Perform a num of iterations of forward/backward
# and optimizations for the sharded module.
num_iters = 10
rank_log(_rank, logger, "Sequence Parallel training starting...")

for i in range(num_iters):
    # For SP, input can be different across all ranks.
    inp = torch.rand(20, 10, device="cuda")
    output = sp_model(inp)
    output.sum().backward()
    optimizer.step()
    rank_log(_rank, logger, f"Sequence Parallel iter {i} completed")

rank_log(_rank, logger, "Sequence Parallel training completed!")
