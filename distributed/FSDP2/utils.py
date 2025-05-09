import torch
from model import Transformer
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import Shard


def inspect_model(model: FSDPModule):
    assert isinstance(model, Transformer)
    assert isinstance(model, FSDPModule)

    if torch.distributed.get_rank() == 0:
        print(model)

    for param in model.parameters():
        assert param.placements == (Shard(0),)
        assert param.dtype == torch.float32
        # print(param.get_local_tensor())


def inspect_mixed_precision(model: FSDPModule):
    model.unshard()
    for param in model.parameters(recurse=False):
        assert param.dtype == torch.bfloat16
    model.reshard()
