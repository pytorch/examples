import os
import argparse
import time
import torch
import torch.nn as nn
from typing import Dict, Any
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    _init_optim_state,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from utils import get_latest_checkpoint_folder
from model import ModelArgs, Transformer


CHECKPOINT_FOLDER = "checkpoints/{api}/{time}" 
MODEL_CHECKPOINT = "model_state_dict.pt"
OPTIM_CHECKPOINT = "optim_state_dict.pt"

def load_from_full_model_state_dict(
    model: FSDPModule,
    full_sd: Dict[str, Any],
    dcp_api: bool,
):
    if dcp_api:
        set_model_state_dict(
            model=model,
            model_state_dict=full_sd,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            ),
        )
        return
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    for param_name, full_tensor in full_sd.items():
        sharded_meta_param = meta_sharded_sd.get(param_name)
        sharded_tensor = distribute_tensor(
            full_tensor,
            sharded_meta_param.device_mesh,
            sharded_meta_param.placements,
        )
        sharded_sd[param_name] = nn.Parameter(sharded_tensor)
    # choose `assign=True` since we cannot call `copy_` on meta tensor
    model.load_state_dict(sharded_sd, strict=False, assign=True)


def get_full_model_state_dict(
    model: FSDPModule,
    dcp_api: bool,
):
    if dcp_api:
        return get_model_state_dict(
            model=model, 
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            )
        )

    sharded_sd = model.state_dict()
    cpu_state_dict = {}
    for param_name, sharded_param in sharded_sd.items():
        full_param = sharded_param.full_tensor()
        if torch.distributed.get_rank() == 0:
            cpu_state_dict[param_name] = full_param.cpu()
        else:
            del full_param
    return cpu_state_dict



def load_from_full_optimizer_state_dict(
    model: FSDPModule,
    opt: torch.optim.Optimizer,
    full_sd: Dict[str, Any],
    dcp_api: bool,
):
    if dcp_api:
        set_optimizer_state_dict(
            model=model, 
            optimizers=opt, 
            optim_state_dict=full_sd, 
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            )
        )
    PARAMS = "params" 
    _init_optim_state(opt)
    param_groups = opt.state_dict()["param_groups"]
    state = opt.state_dict()["state"]

    full_param_groups = full_sd["param_groups"]
    full_state = full_sd["state"]

    for param_group, full_param_group in zip(param_groups, full_param_groups):
        for key, value in full_param_group.items():
            if key == PARAMS:
                continue
            param_group[key] = value
        for pid, full_pid in zip(param_group[PARAMS], full_param_group[PARAMS]):
            if pid not in state:
                continue
            param_state = state[pid]
            full_param_state = full_state[full_pid]
            for attr, full_tensor in full_param_state.items():
                sharded_tensor = param_state[attr]
                if isinstance(sharded_tensor, DTensor):
                    # exp_avg is DTensor
                    param_state[attr] = distribute_tensor(
                        full_tensor,
                        sharded_tensor.device_mesh,
                        sharded_tensor.placements,
                    )
                else:
                    # step is plain tensor
                    param_state[attr] = full_tensor
    opt.load_state_dict(
        {
            "param_groups": param_groups,
            "state": state,
        }
    )


def get_full_optimizer_state_dict(
    model: FSDPModule, 
    opt: torch.optim.Optimizer,
    dcp_api: bool
):
    if dcp_api:
        return get_optimizer_state_dict(
            model=model, 
            optimizers=opt, 
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            )
        )
    is_rank_zero = (torch.distributed.get_rank() == 0)
    sharded_sd = opt.state_dict()
    sharded_state = sharded_sd["state"]
    full_state = {}
    for group_id, sharded_group in sharded_state.items():
        group_state = {}
        for attr, sharded_tensor in sharded_group.items():
            if isinstance(sharded_tensor, DTensor):
                # "exp_avg" in AdamW is `DTensor`
                full_tensor = sharded_tensor.full_tensor()
            else:
                # "step" in AdamW is plain tensor
                full_tensor = sharded_tensor
            if is_rank_zero:
                group_state[attr] = full_tensor.cpu()
            else:
                del full_tensor
        if is_rank_zero:
            full_state[group_id] = group_state
        else:
            del group_state
    if is_rank_zero:
        return {
            "param_groups": sharded_sd["param_groups"],
            "state": full_state,
        }
    else:
        return {}


def inspect_model(model: FSDPModule):
    assert isinstance(model, Transformer)
    assert isinstance(model, FSDPModule)

    if torch.distributed.get_rank() == 0:
        print(model)

    # inspect model parameters
    for param in model.parameters():
        assert param.placements == (Shard(0),)
        assert param.dtype  == torch.float32
        # print(param.get_local_tensor())

def inspect_mixed_precision(model: FSDPModule):
    model.unshard()
    for param in model.parameters(recurse=False):
        assert param.dtype == torch.bfloat16
    model.reshard()

def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)

def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)

def main(args):
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.manual_seed(0)
    vocab_size = 1024
    batch_size = 32
    seq_len = 64
    model_args = ModelArgs(
        n_layers=10,
        n_heads=4,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        dropout_p=0,
    )
    with torch.device("meta"):
        model = Transformer(model_args)
    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, 
            reduce_dtype=torch.float32,
        )
    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    last_training_time = get_latest_checkpoint_folder(f"checkpoint/{'dcp_api' if args.dcp_api else 'dtensor_api'}")

    if last_training_time is None:
        model.to_empty(device="cuda")
        model.reset_parameters()
    else:
        last_model_checkpoint = CHECKPOINT_FOLDER.format(
            api="dcp_api" if args.dcp_api else "dtensor_api", 
            time=last_training_time,
        ) + f"/{MODEL_CHECKPOINT}"
        model_state_dict = torch.load(last_model_checkpoint, mmap=True, weights_only=True, map_location='cpu')
        load_from_full_model_state_dict(model, model_state_dict, dcp_api=args.dcp_api)

    inspect_model(model)
    if args.mixed_precision:
        inspect_mixed_precision(model)

    if args.explicit_prefetching:
        set_modules_to_forward_prefetch(model, num_to_forward_prefetch=2)
        set_modules_to_backward_prefetch(model, num_to_backward_prefetch=2)
    
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    if last_training_time is not None:
        last_optim_checkpoint = CHECKPOINT_FOLDER.format(
            api="dcp_api" if args.dcp_api else "dtensor_api", 
            time=last_training_time,
        ) + f"/{OPTIM_CHECKPOINT}"
        optim_state_dict = torch.load(last_optim_checkpoint, mmap=True, weights_only=True, map_location='cpu')
        load_from_full_optimizer_state_dict(model, optim, optim_state_dict, dcp_api=args.dcp_api)
    torch.distributed.barrier()
    for _ in range(10):
        if args.explicit_prefetching:
            model.unshard()
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        loss = model(x).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad()
    
    model_state_dict = get_full_model_state_dict(model, dcp_api=args.dcp_api)
    optim_state_dict = get_full_optimizer_state_dict(model, optim, dcp_api=args.dcp_api)

    if torch.distributed.get_rank() == 0:
        new_training_time = current_time_ms = int(time.time() * 1000)
        new_checkpoint_folder = CHECKPOINT_FOLDER.format(
            api="dcp_api" if args.dcp_api else "dtensor_api", 
            time=new_training_time,
        )
        new_model_checkpoint = f"{new_checkpoint_folder}/{MODEL_CHECKPOINT}"
        new_optim_checkpoint = f"{new_checkpoint_folder}/{OPTIM_CHECKPOINT}"
        os.makedirs(new_checkpoint_folder, exist_ok=True)
        torch.save(model_state_dict, new_model_checkpoint)
        torch.save(optim_state_dict, new_optim_checkpoint)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch FSDP2 example')
    parser.add_argument('--explicit-prefetching', action='store_true', default=False)
    parser.add_argument('--mixed-precision', action='store_true', default=False)
    parser.add_argument('--dcp-api', action="store_true", default=False)
    args = parser.parse_args()
    main(args)
