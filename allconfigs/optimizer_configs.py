import dataclasses
from typing import Tuple
from dataclasses import dataclass

@dataclass
class OptimizerConf:
    name: str = dataclasses.field(init=False)

@dataclass
class AdamConf(OptimizerConf):
    name: str = 'adam'
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False

@dataclass
class SGDConf(OptimizerConf):
    name: str = 'sgd'
    momentum: float = 0.9
    weight_decay: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False

@dataclass
class LRSchedulerConf:
    name: str = dataclasses.field(init=False)

@dataclass
class CosineSchedulerConf(LRSchedulerConf):
    name: str = 'cosine'
    T_max: int = 50  # Maximum number of iterations
    eta_min: float = 1e-5  # Minimum learning rate
    last_epoch: int = -1  # Last epoch for restart

@dataclass
class LinearSchedulerConf(LRSchedulerConf):
    name: str = 'linear'
    total_iters: int = 100  # Number of iterations for decay
    start_factor: float = 1.0  # Starting factor of LR
    end_factor: float = 0.0  # Ending factor of LR

@dataclass
class NoSchedulerConf(LRSchedulerConf):
    name: str = 'none'

@dataclass
class WarmupConf:
    use_warmup: bool = False
    epochs: int = 5
    start_factor: float = 1e-5
    warmup_strategy: str = "linear"