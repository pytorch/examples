
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)




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


class ToyModel(nn.Module):
    """ MLP based model """
    def __init__(self):
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(32, 5)

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))

"""
Main body of the demo of a basic version of tensor parallel by using
PyTorch native APIs.
"""

# understand world topology
_rank = int(os.environ["RANK"])
_local_rank = int(os.environ["LOCAL_RANK"])
_world_size = int(os.environ["WORLD_SIZE"])
#_local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])



def rank_print(msg):
    """helper function to print only on global rank 0"""
    if _rank==0:
        print(f"{msg}")

print(f"Running basic Megatron style TP example on rank {_rank}.")
assert _world_size % 2 == 0, f"TP examples require even number of GPUs, but got {_world_size} gpus"



# create a device mesh based on the given world_size.

_device = f"cuda"
device_mesh = init_device_mesh(device_type = _device,mesh_shape = (_world_size,))
assert device_mesh is not None, "unable to create valid device mesh"

rank_print(f"Device Mesh created: {device_mesh=}")

# create model and move it to GPU - init_device_mesh has already mapped GPU ids.
tp_model = ToyModel().to(_device)

# Create an optimizer for the parallelized module.
lr = 0.25
optimizer = torch.optim.AdamW(tp_model.parameters(), lr=lr, foreach=True)

# Custom parallelization plan for the model
tp_model = parallelize_module(module = tp_model,
                                device_mesh = device_mesh,
                                parallelize_plan = {
                                    "in_proj": ColwiseParallel(),
                                    "out_proj": RowwiseParallel(),
                                },
)
# Perform a num of iterations of forward/backward
# and optimizations for the sharded module.
num_iters = 10
rank_print(f"Tensor Parallel training starting...")

for i in range(num_iters):
    # For TP, input needs to be same across all TP ranks.
    # Setting the random seed is to mimic the behavior of dataloader.
    torch.manual_seed(i)
    inp = torch.rand(20, 10, device=_device)
    output = tp_model(inp)
    output.sum().backward()
    optimizer.step()
    rank_print(f"Tensor Parallel iter {i} completed")

rank_print(f"Tensor Parallel training completed!")
