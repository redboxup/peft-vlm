from dataclasses import dataclass

@dataclass
class TuneConf:
    num_samples: int = 1
    gpus_per_trial: float = 1
    cpus_per_trial: int = 16