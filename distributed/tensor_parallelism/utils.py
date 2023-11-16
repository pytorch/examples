import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12359"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def torchrun_setup():
    """we use torchrun for init so no params needed here"""
    dist.init_process_group("nccl")

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

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

class MLP_swiglu(nn.Module):
    def __init__(self, mlp_dim: int= 1024) -> None:
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
