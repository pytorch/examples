from dataclasses import dataclass
from typing import ClassVar


@dataclass
class train_config:
    model_name: str="t5-base"
    run_validation: bool=True
    batch_size_training: int=4
    num_workers_dataloader: int=2
    lr: float=0.002
    weight_decay: float=0.0
    gamma: float= 0.85
    use_fp16: bool=False
    mixed_precision: bool=True
    save_model: bool=False
    
    
    