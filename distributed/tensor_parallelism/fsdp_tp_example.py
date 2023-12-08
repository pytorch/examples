import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

import os
from log_utils import rank_log, get_logger, verify_min_gpu_count


# ---- GPU check ------------
_min_gpu_count = 4

if not verify_min_gpu_count(min_gpus=_min_gpu_count):
    print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
    sys.exit()
# ---------------------------

from torch.distributed._tensor.device_mesh import init_device_mesh


"""
This is the script to test 2D Parallel which combines Tensor/Sequence
parallel with Fully Sharded Data Parallel (TP/SP + FSDP) on a toy model
in the SPMD style. We show an E2E working flow from forward, backward
and optimization.

We enabled Fully Sharded Data Parallel + Tensor Parallel in
separate parallel dimensions:
    Data Parallel ("dp") across hosts
    Tensor Parallel ("tp") within each host

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


def find_multiple(n: int, k: int) -> int:
    """function to find resizing multiple for SwiGLU MLP"""
    if n % k == 0:
        return n
    return n + k - (n % k)


class MLP_swiglu(nn.Module):
    """SwiGLU to showcase a Llama style MLP model"""

    def __init__(self, mlp_dim: int = 1024) -> None:
        super().__init__()
        hidden_dim = 4 * mlp_dim
        scaled_hidden = int(2 * hidden_dim / 3)
        rounded_hidden = find_multiple(scaled_hidden, 256)

        self.in_proj = nn.Linear(mlp_dim, rounded_hidden, bias=False)
        self.gate_proj = nn.Linear(mlp_dim, rounded_hidden, bias=False)
        self.out_proj = nn.Linear(rounded_hidden, mlp_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.in_proj(x)) * self.gate_proj(x)
        x = self.out_proj(x)
        return x


"""
Main body of the demo of a basic version of tensor parallel by using
PyTorch native APIs.
"""
tp_size = 2
logger = get_logger()

# understand world topology
_rank = int(os.environ["RANK"])
_world_size = int(os.environ["WORLD_SIZE"])


print(f"Starting PyTorch 2D (FSDP + TP) example on rank {_rank}.")
assert (
    _world_size % tp_size == 0
), f"World size {_world_size} needs to be divisible by TP size {tp_size}"


# create a sharding plan based on the given world_size.
dp_size = _world_size // tp_size

# Create a device mesh with 2 dimensions.
# First dim is the data parallel dimension
# Second dim is the tensor parallel dimension.
device_mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")
tp_mesh = device_mesh["tp"]
dp_mesh = device_mesh["dp"]

# For TP, input needs to be same across all TP ranks.
# while for SP, input can be different across all ranks.
# We will use dp_rank for setting the random seed
# to mimic the behavior of the dataloader.
dp_rank = dp_mesh.get_local_rank()

# create model and move it to GPU with id rank
_mlp_dim = 1024
base_model_swiglu = MLP_swiglu(mlp_dim=_mlp_dim).to("cuda")


# Custom parallelization plan for the swiglu MLP model
custom_tp_model = parallelize_module(
    module=base_model_swiglu,
    device_mesh=tp_mesh,
    parallelize_plan={
        "in_proj": ColwiseParallel(),
        "gate_proj": ColwiseParallel(),
        "out_proj": RowwiseParallel(),
    },
)

rank_log(_rank, logger, f"Model after parallelization {custom_tp_model=}\n")

# Init FSDP using the dp device mesh
sharded_model = FSDP(custom_tp_model, device_mesh=dp_mesh, use_orig_params=True)

# Create an optimizer for the parallelized and sharded model.
lr = 3e-3
rank_log(_rank, logger, f"Creating AdamW optimizer with learning rate {lr}")
optimizer = torch.optim.AdamW(sharded_model.parameters(), lr=lr, foreach=True)

# Training loop:
# Perform a num of iterations of forward/backward
# and optimizations for the sharded module.
rank_log(_rank, logger, "\nStarting 2D training...")
num_iterations = 10
batch_size = 2

for i in range(num_iterations):
    # seeding with dp_rank to ensure identical inputs for TP groups
    torch.manual_seed(i + dp_rank)
    inp = torch.rand(batch_size, _mlp_dim, device="cuda")

    output = sharded_model(inp)
    output.sum().backward()
    optimizer.step()
    rank_log(_rank, logger, f"2D iter {i} complete")

rank_log(_rank, logger, "2D training successfully completed!")
