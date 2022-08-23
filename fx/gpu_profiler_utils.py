import math
from enum import Enum, auto
from typing import List, Tuple

import tabulate
import torch
import torch.fx as fx
import torch.utils._pytree as pytree

from torch.fx import GraphModule, Interpreter, Node

# some pytorch low-level memory management constant
# the minimal allocate memory size (Byte)
PYTORCH_MIN_ALLOCATE = 2**20
# the minimal cache memory size (Byte)
PYTORCH_MIN_CACHE = 2**20
#default device for graph based profiling
DEVICE = torch.device("cuda")
#Used for determining if the peak memory usage exceeds the device memory
MEM_LIMIT = torch.cuda.get_device_properties(DEVICE).total_memory


class GraphProfiler(Interpreter):
    pass
#Bidirectional Dictionary to store the mapping 
class BiDict(dict):
    def __init__(self, *args, **kwargs):
        super(BiDict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(BiDict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(BiDict, self).__delitem__(key)


class GraphType(Enum):
    forward = auto()
    backward = auto()


class ProfileMode(Enum):
    default = auto()
    swap = auto()
    mem_saver_swap = auto()


profile_mode_dict = {
    "default": ProfileMode.default,
    "swap": ProfileMode.swap,
    "mem_saver_swap": ProfileMode.mem_saver_swap,
}


class TensorStatus(Enum):
    cpu = auto()
    gpu = auto()
    deleted = auto()
    recomputed = auto()


def same_storage(x: torch.Tensor, y: torch.Tensor) -> bool:
    return x.storage().data_ptr() == y.storage().data_ptr()


def get_tensor_stat(tensor: torch.Tensor) -> Tuple[int, int, int]:

    if tensor.is_sparse:
        indices_stat = get_tensor_stat(tensor._indices())
        values_stat = get_tensor_stat(tensor._values())
        tensor = None
        return indices_stat + values_stat

    numel = tensor.numel()
    element_size = tensor.storage().element_size()
    fact_numel = tensor.storage().size()
    fact_memory_size = fact_numel * element_size
    # since pytorch allocate at least 512 Bytes for any tensor, round
    # up to a multiple of 512
    memory_size = (
        math.ceil(fact_memory_size / PYTORCH_MIN_ALLOCATE) * PYTORCH_MIN_ALLOCATE
    )
    size = tuple(tensor.size())
    # torch scalar has empty size
    if not size:
        size = (1,)
    tensor = None
    return (size, numel, memory_size)


# Node Information on All Tensors produced in a graph
class NodeInfo:
    def __init__(self):
        self.rank: int = 0
        self.gtype: GraphType = None
        # populated during profiling
        self.run_time: float = 1
        self.cumulative_run_time: float = 1
        self.peak_mem: int = 0
        self.active_mem: int = 0
        self.in_peak_interval: bool = False
        self.total_peak_mem: int = 0
        # populated based on needs
        self.first_forward_access: Node = None
        self.last_forward_access: Node = None
        self.first_back_access: Node = None
        self.last_back_access: Node = None
        # populated during scheduling algorithm (Future use)
        self.fw_nodes: List[Node] = []
        self.fw_backup: List[Node] = []
        self.intermediate_mem: float = 0
        self.to_offload: List[Node] = []
        self.to_prefetch: List[Node] = []
        self.to_recompute: List[Node] = []
        self.to_delete: List[Node] = []


# Node Information for Intermediate Tensors
class IntNodeInfo(NodeInfo):
    def __init__(self):
        super().__init__()
        # populated during profiling
        self.idle_time: float = 0
        self.swap_time: float = 0
        self.size: int = 0
        self.memory_size: int = 0
        self.numel: int = 0
        self.cpu_ref: torch.Tensor = None
        self.status: TensorStatus = TensorStatus.deleted
        # attributed related to swap, populated during scheduling algorithm (Future Use)
        self.prefetch_trigger_start: Node = None
        self.prefetch_trigger_end: Node = None
        self.prefetch_begin: torch.cuda.Event = None
        self.prefetch_end: torch.cuda.Event = None
        self.offload_begin: torch.cuda.Event = None
        self.active_forward_interval: List[Node] = []
        self.active_backward_interval: List[Node] = []
        # attributes related to recomputation only, populated during recomp algorithm (Future Use)
        self.rcomp_sources: List[Node] = None
        self.rcomp_primals: List[Node] = None
        self.rcomp_extra: List[Node] = None
        self.rcomp_graph_mod: GraphModule = None
        self.rcomp_executor: Interpreter = None
        self.exe_count: int = 0
        self.rcomp_time: float = 0
        self.exe_time: float = 0
        self.rcomp_mem: int = 0
        self.MSPS: float = 0
        self.is_recomp: bool = False

    def updateMSPS(self):
        # The metric currently being used in Recomputation algorithm (Future Use)
        self.MSPS = self.memory_size / self.exe_time
