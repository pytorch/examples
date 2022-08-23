import math
from enum import Enum, auto
from typing import List, Tuple

import torch
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

#forward declaration of the GraphProfiler class
class GraphProfiler(Interpreter):
    pass

#Bidirectional Dictionary to store the mapping of the forward and backward pass intermediate nodes
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
    r"""
        ProfileMode : The Graph Profiler provides three profiling modes,``default``, ``swap`` and ``mem_saver_swap``.
                            default: Measure the per node run-time, active memory usage, peak memory usage, marks the intermediate
                                    nodes (activations saved from forward pass and needed in backward pass), registers their last
                                    use in the forward pass and first use in the backward pass, measures this idle time, measures
                                    their memory size and, marks the first use of the model parameter nodes in the forward pass.
                            swap:   All the of the above plus measures the time to swap each of the intermediate tensors (activations)
                                     to CPU memory, back and forth.
                            mem_saver_swap: All of the above and profiles in a low memory mode, pushing all of the activations to
                                            the CPU memory during the forward pass and fetches them back when they are needed in
                                            the backward pass. Allows profiling graphs way larger than GPU memory.
    """
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
    r"""
    Utility method that provides stats on the queried tensor.
    Args: 
        tensor (torch.Tensor): Input tensor to get that stats for
    Returns:
        Tuple(size, numel, memory_size): 
            size: the dimensions of ``tensor``
            numel: number of elements in the ``tensor``
            memory_size: the physical memeory occupied by the ``tensor`` in bytes.
    """

    if tensor.is_sparse:
        indices_stat = get_tensor_stat(tensor._indices())
        values_stat = get_tensor_stat(tensor._values())
        tensor = None
        return indices_stat + values_stat

    numel = tensor.numel()
    element_size = tensor.storage().element_size()
    fact_numel = tensor.storage().size()
    fact_memory_size = fact_numel * element_size
    # rounding up to pytorch's allocation granularity
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
    r"""
    The base class to store the profiling and static graph analysis information for all the nodes in the graph.
    1) rank (int): stores the rank of the node in the order that is executed.
    2) gtype (GraphType): denotes if the node belongs to the forward/backward graph
    3) run_time (float): the recorded run-time to the node in ms.
    4) cumulative_run_time (float): the cumulative run-time of the node from the first node in the graph.
    5) peak_mem (int): the peak memory usage of the node during execution in bytes.
    6) active_mem (int): the number of bytes active subsequent to the node's execution.
    7) in_peak_interval (bool): flags is True if the node lies in the peak memory interval (when active memory 
                                exceeds device memory limit)
    8) total_peak_mem (int): This is the peak memory consumption calculated by simulating the memory usage for 
                            low memory mode.
    9) first_forward_access (Node): Reference to the node that first uses self in the forward pass.
                                    Generally populated for the parameter nodes of the model.
    10) last_forward_access (Node): Reference to the node that last uses self in the forward pass.
                                    Generally populated for the intermediate (activation) nodes.
    11) first_back_access (Node): Reference to the node that first uses self in the backward pass.
                                    Generally populated for the intermediate (activation) nodes.
    12) last_back_access (Node): Reference to the node that last uses self in the backward pass.
                                    Generally populated for the intermediate (activation) nodes.
    """
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
    r"""
    Derieved class to store the profiling and static graph analysis information for intermediate 
    nodes (activations) in the graph.
    1) idle_time (float): The idle time is calculated as [(last_backward_acess - swap_time) - (last_forward_access + swap_time)].
    2) swap_time (float): The time in ms required to swap tensor to and fro CPU memory.
    3) size (Tuple): The dimension of the intermediate tensor. 
    4) memory_size (int): the physical memeory occupied by the tensor in bytes
    5) numel (int): number of elements in the tensor.
    6) cpu_ref (torch.Tensor): The reference to the pinned memory CPU tensor.
    7) status (TensorStatus): Current status of the tensor (CPU/GPU/Deleted)
    """
    def __init__(self):
        super().__init__()
        # populated during profiling
        self.idle_time: float = 0
        self.swap_time: float = 0
        self.size: Tuple = 0
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
