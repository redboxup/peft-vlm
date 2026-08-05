import hydra
import dataclasses
from pathlib import Path
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

from allconfigs.tuner_configs import TuneConf

from allconfigs.optimizer_configs import (
    OptimizerConf, LRSchedulerConf, AdamConf, SGDConf, CosineSchedulerConf,
    NoSchedulerConf, LinearSchedulerConf, WarmupConf,
)

from allconfigs.train_configs import (
    TrainConf, LoCoOpConf, ZeroShotConf, CoOpConf, MoCoOpConf, DualCoOpConf,
    DualCoOpPPConf, PLOTConf, GalLoPConf, ManualPromptConf, Ensemble_PromptConf1, Vision_ClassifierConf,
    Ensemble_PromptConf2, Ensemble_PromptConf3,
)

from allconfigs.eval_configs import (
    EvalConf, MCMConf, GLMCMConf, LMCMConf, MCM_DualCoOpConf,
    LMCM_DualCoOpConf, GLMCM_DualCoOpConf, EnergyV1Conf, EnergyV2Conf
)

from allconfigs.dataset_configs import (
    DatasetConf, CheXpertConf, VinCXRConf, NIHCXRConf, OODDatasetConf,
)


@dataclass
class ProjectConf:
    trainer: TrainConf
    dataset: DatasetConf
    optimizer: OptimizerConf = dataclasses.field(
        default_factory=AdamConf
    )
    scheduler: LRSchedulerConf = dataclasses.field(
        default_factory=NoSchedulerConf
    )
    eval: EvalConf = EvalConf  # type: ignore
    ood: OODDatasetConf = OODDatasetConf  # type: ignore
    warmup: WarmupConf = WarmupConf  # type: ignore
    tune: TuneConf = TuneConf  # type: ignore
    device: str = 'cuda:0'
    multigpu: bool = False
    model: str = 'ViT-B/32'
    rootpath: Path = dataclasses.field(init=False)
    experiment_dir: Path = dataclasses.field(init=False)
    weights_dir: str = 'weights/chexzero_weights/CheXzero_Models'
    verbose: bool = True


cs = ConfigStore.instance()
cs.store(name='projectconf', node=ProjectConf)
cs.store(name='coop', group='trainer', node=CoOpConf)
cs.store(name='plot', group='trainer', node=PLOTConf)
cs.store(name='mocoop', group='trainer', node=MoCoOpConf)
cs.store(name='locoop', group='trainer', node=LoCoOpConf)
cs.store(name='gallop', group='trainer', node=GalLoPConf)
cs.store(name='zeroshot', group='trainer', node=ZeroShotConf)
cs.store(name='manual_prompt', group='trainer', node=ManualPromptConf)
cs.store(name='dualcoop', group='trainer', node=DualCoOpConf)
cs.store(name='dualcoop++', group='trainer', node=DualCoOpPPConf)
cs.store(name='ensemble_prompt1', group='trainer', node=Ensemble_PromptConf1)
cs.store(name='vision_classifier', group = 'trainer', node=Vision_ClassifierConf )
cs.store(name='ensemble_prompt2', group='trainer', node=Ensemble_PromptConf2)
cs.store(name='ensemble_prompt3', group='trainer', node=Ensemble_PromptConf3)
cs.store(name='mcm', group='eval', node=MCMConf)
cs.store(name='lmcm', group='eval', node=LMCMConf)
cs.store(name='glmcm', group='eval', node=GLMCMConf)
cs.store(name='energy_v1', group='eval', node=EnergyV1Conf)
cs.store(name='energy_v2', group='eval', node=EnergyV2Conf)
cs.store(name='mcm_dualcoop', group='eval', node=MCM_DualCoOpConf)
cs.store(name='lmcm_dualcoop', group='eval', node=LMCM_DualCoOpConf)
cs.store(name='glmcm_dualcoop', group='eval', node=GLMCM_DualCoOpConf)
cs.store(name='sgd', group='optimizer', node=SGDConf)
cs.store(name='adam', group='optimizer', node=AdamConf)
cs.store(name='cosine', group='scheduler', node=CosineSchedulerConf)
cs.store(name='linear', group='scheduler', node=LinearSchedulerConf)
cs.store(name='nihcxr', group='dataset', node=NIHCXRConf)
cs.store(name='vincxr', group='dataset', node=VinCXRConf)
cs.store(name='chexpert', group='dataset', node=CheXpertConf)


@hydra.main(config_name='projectconf', version_base='1.3')
def main(config: ProjectConf):
    print(config)


if __name__ == '__main__':
    main()
