import os
import time

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    _init_optim_state,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import distribute_tensor, DTensor


MODEL_CHECKPOINT = "model_state_dict.pt"
OPTIM_CHECKPOINT = "optim_state_dict.pt"
PARAMS = "params"


def get_latest_checkpoint_folder(path):
    max_num = None
    if not os.path.exists(path):
        return max_num
    for name in os.listdir(path):
        folder_path = os.path.join(path, name)
        if os.path.isdir(folder_path):
            try:
                num = int(name)
                if max_num is None or num > max_num:
                    max_num = num
            except ValueError:
                pass  # Skip non-numeric folder names
    return max_num


class Checkpointer:
    def __init__(self, folder: str, dcp_api: bool):
        self.folder = folder
        self.dcp_api = dcp_api
        self.last_training_time = get_latest_checkpoint_folder(
            f"{folder}/{'dcp_api' if dcp_api else 'dtensor_api'}"
        )

    def is_empty(self):
        return self.last_training_time is None

    def load_model(self, model: FSDPModule):
        last_model_checkpoint = (
            f"{self.folder}/{'dcp_api' if self.dcp_api else 'dtensor_api'}"
            f"/{self.last_training_time}/{MODEL_CHECKPOINT}"
        )
        full_sd = torch.load(
            last_model_checkpoint, mmap=True, weights_only=True, map_location="cpu"
        )
        if self.dcp_api:
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

    def load_optim(self, model: FSDPModule, opt: torch.optim.Optimizer):
        last_optim_checkpoint = (
            f"{self.folder}/{'dcp_api' if self.dcp_api else 'dtensor_api'}"
            f"/{self.last_training_time}/{OPTIM_CHECKPOINT}"
        )
        full_sd = torch.load(
            last_optim_checkpoint, mmap=True, weights_only=True, map_location="cpu"
        )
        if self.dcp_api:
            set_optimizer_state_dict(
                model=model,
                optimizers=opt,
                optim_state_dict=full_sd,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                ),
            )
            return
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

    def _get_full_model_state_dict(self, model: FSDPModule):
        if self.dcp_api:
            return get_model_state_dict(
                model=model,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
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

    def _get_full_optimizer_state_dict(
        self,
        model: FSDPModule,
        opt: torch.optim.Optimizer,
    ):
        if self.dcp_api:
            return get_optimizer_state_dict(
                model=model,
                optimizers=opt,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )
        is_rank_zero = torch.distributed.get_rank() == 0
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

    def save(self, model: FSDPModule, optim: torch.optim.Optimizer):
        model_state_dict = self._get_full_model_state_dict(model)
        optim_state_dict = self._get_full_optimizer_state_dict(model, optim)
        if torch.distributed.get_rank() == 0:
            new_training_time = int(time.time() * 1000)
            new_checkpoint_folder = f"{self.folder}/{'dcp_api' if self.dcp_api else 'dtensor_api'}/{new_training_time}"
            new_model_checkpoint = f"{new_checkpoint_folder}/{MODEL_CHECKPOINT}"
            new_optim_checkpoint = f"{new_checkpoint_folder}/{OPTIM_CHECKPOINT}"
            os.makedirs(new_checkpoint_folder, exist_ok=True)
            torch.save(model_state_dict, new_model_checkpoint)
            torch.save(optim_state_dict, new_optim_checkpoint)
