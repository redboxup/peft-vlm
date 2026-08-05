import dataclasses
from typing import Tuple
from dataclasses import dataclass

@dataclass
class EvalConf:
    name: str = dataclasses.field(init=False)
    batch_size: int = 32
    evaluate_over: Tuple[str, ...] = ('mura', 'irma', 'boneage')
    num_workers: int = 0

    # method specific parameters
    lambda_lmcm: float = dataclasses.field(init=False)

    # evaluate top-k over sweep
    top_k: int = 7

@dataclass
class MCMConf(EvalConf):
    name: str = 'mcm'

@dataclass
class LMCMConf(EvalConf):
    name: str = 'lmcm'

@dataclass
class GLMCMConf(EvalConf):
    name: str = 'glmcm'
    lambda_lmcm: float = 0.25

@dataclass
class MCM_DualCoOpConf(EvalConf):
    name: str = 'mcm_dualcoop'

@dataclass
class LMCM_DualCoOpConf(EvalConf):
    name: str = 'lmcm_dualcoop'

@dataclass
class GLMCM_DualCoOpConf(EvalConf):
    name: str = 'glmcm_dualcoop'
    lambda_lmcm: float = 0.25

@dataclass
class EnergyV1Conf(EvalConf):
    name: str = 'energy_v1'

@dataclass
class EnergyV2Conf(EvalConf):
    name: str = 'energy_v2'