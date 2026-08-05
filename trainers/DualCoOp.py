import torch
from logging import Logger
from typing import Dict, Any
from rich.console import Console
from torch.utils.data import DataLoader
from torch.optim import optimizer, lr_scheduler

from configs import ProjectConf
from trainers import BaseTrainer
from trainers import CoOpTrainer
from models.DualCoOp import DualCoOpCLIP
from losses.DualCoOp import AsymmetricLoss


@BaseTrainer.register_trainer('dualcoop')
class DualCoOpTrainer(CoOpTrainer):
    def __init__(
        self,
        configs: ProjectConf,
        model: DualCoOpCLIP,
        transforms: Dict[str, Any],
        dataloaders: Dict[str, DataLoader],
        optimizer: optimizer.Optimizer,
        scheduler: lr_scheduler.LRScheduler,
        logger: Logger
    ):
        BaseTrainer.__init__(
            self,
            configs,
            model,
            transforms,
            dataloaders,
            optimizer,
            scheduler,
            logger,
        )
        self.model: DualCoOpCLIP  # type: ignore
        self.dataloaders: Dict[str, DataLoader]
        self.console = Console()
        self.logger = logger
        self.freeze_model()

    def get_losses(self) -> Dict[str, Any]:
        return dict(
            As=AsymmetricLoss(
                gamma_pos=self.configs.trainer.gamma_pos,  # type: ignore
                gamma_neg=self.configs.trainer.gamma_neg,  # type: ignore
            )
        )

    def model_forward(self, batch_input) -> torch.Tensor:  # type: ignore
        images = self.transforms['train'](
            batch_input['img'].to(self.device)
        )
        output = self.model.forward_with_neg(images)
        return output

    def compute_loss(self, loss_name, loss_fn, batch_input, outputs):
        labels = batch_input['labels'].to(self.device)
        if loss_name == 'As':
            return loss_fn(outputs, labels)
