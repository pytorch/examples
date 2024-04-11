import sys
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from log_utils import rank_log, get_logger, verify_min_gpu_count

# ---- GPU check ------------
_min_gpu_count = 4

if not verify_min_gpu_count(min_gpus=_min_gpu_count):
    print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
    sys.exit()
# ---------------------------

from llama2_model import Transformer, ModelArgs

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel
)


"""
This is the script to test 2D Parallel which combines Tensor/Sequence
parallel with Fully Sharded Data Parallel (TP/SP + FSDP) on a example
Llama2 model. We show an E2E working flow from forward, backward
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

More details can be seen in the PyTorch tutorials:
https://pytorch.org/tutorials/intermediate/TP_tutorial.html
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

# create model and move it to GPU - init"cuda"_mesh has already mapped GPU ids.
simple_llama2_config = ModelArgs(dim=256, n_layers=2, n_heads=16, vocab_size=32000)

model = Transformer.from_model_args(simple_llama2_config).to("cuda")

# init model weights
model.init_weights()

# parallelize the first embedding and the last linear out projection
model = parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
        ),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Replicate()
        ),
        "norm": SequenceParallel(),
        "layers.0": PrepareModuleInput(
            input_layouts=(Replicate(), None),
            desired_input_layouts=(Shard(1), None),
            use_local_output=True,
        ),
    }
)

for layer_id, transformer_block in enumerate(model.layers):
    layer_tp_plan = {
        "attention": PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        ),
        "attention.wq": ColwiseParallel(),
        "attention.wk": ColwiseParallel(),
        "attention.wv": ColwiseParallel(),
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
        "attention_norm": SequenceParallel(),
        "feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        "feed_forward.w3": ColwiseParallel(),
        "ffn_norm": SequenceParallel(),
    }

    # Adjust attention module to use the local number of heads
    attn_layer = transformer_block.attention
    attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
    attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()

    # Custom parallelization plan for the model
    parallelize_module(
        module=transformer_block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_tp_plan
    )

# Init FSDP using the dp device mesh
sharded_model = FSDP(model, device_mesh=dp_mesh, use_orig_params=True)

rank_log(_rank, logger, f"Model after parallelization {sharded_model=}\n")

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
    inp = torch.randint(32000, (8, 256), device="cuda")

    output = sharded_model(inp)
    output.sum().backward()
    optimizer.step()
    rank_log(_rank, logger, f"2D iter {i} complete")

rank_log(_rank, logger, "2D training successfully completed!")
