from .optimizer_configs import (
    AdamConf, SGDConf, CosineSchedulerConf, LinearSchedulerConf,
    NoSchedulerConf)

CONFIG_REGISTRY = {
    'optimizer' : {
        'adam': AdamConf,
        'sgd': SGDConf,
    },
    'scheduler': {
        'cosine': CosineSchedulerConf,
        'linear': LinearSchedulerConf,
        'none': NoSchedulerConf,
    }
}