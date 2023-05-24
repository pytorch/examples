from dataclasses import dataclass, field
from typing import ClassVar
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

@dataclass
class fsdp_config:
    mixed_precision: bool=True
    use_fp16: bool=False
    seed: int=42
    fsdp_activation_checkpointing: bool=True
    limit_all_gathers: bool=True
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD #HYBRID_SHARD, SHARD_GRAD_OP
    checkpoint_type: StateDictType = StateDictType.FULL_STATE_DICT # alternatively can use SHARDED_STATE_DICT to avoid OOMs
    save_optimizer: bool=False
    
    
    
    