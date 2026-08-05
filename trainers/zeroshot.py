import torch
import numpy as np
import pandas as pd
from logging import Logger
from typing import Dict, Any
from rich.console import Console
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import optimizer, lr_scheduler

from models.zeroshot import ZeroShotCLIP
from trainers import BaseTrainer
from configs import ProjectConf


@BaseTrainer.register_trainer('zeroshot')
class ZeroShotTrainer(BaseTrainer):
    def __init__(
        self,
        configs: ProjectConf,
        model: ZeroShotCLIP,
        transforms: Dict[str, Any],
        dataloaders: Dict[str, DataLoader],
        optimizer: optimizer.Optimizer,
        scheduler: lr_scheduler.LRScheduler,
        logger: Logger
    ):
        super().__init__(
            configs, model, transforms, dataloaders, optimizer, scheduler, logger
        )
        self.model: ZeroShotCLIP
        self.dataloaders: Dict[str, DataLoader]
        self.logger = logger
        self.console = Console()
        self.logger.info(
            "[bold green]ZeroShotTrainer initialized.[/bold green]"
        )

    def get_losses(self):
        return {}

    def train(self):
        self.save_checkpoint(epoch=0, best=True)
        self.post_train_epoch()
        self.logger.info(
            "No Training Necessary for Zero-Shot Learning."
        )

    def forward_pass(self, split: str):  # type: ignore
        logits_list, negative_logits_list, labels_list = [], [], []

        with torch.inference_mode():
            for data in self.tqdm(
                self.dataloaders[split], desc=f'[{split}] Forward Pass'
            ):
                images, labels = data['img'].to(self.device), data['labels']
                logits, negative_logits = self.model(images)

                logits_list.extend(logits.cpu())
                negative_logits_list.extend(negative_logits.cpu())
                labels_list.extend(labels.numpy())

        return logits_list, negative_logits_list, labels_list

    def evaluate(self, split: str) -> pd.DataFrame:
        self.logger.info("Evaluate on Test Data")
        self.model.eval()

        logits_list, negative_logits_list, labels_list = self.forward_pass(
            split=split
        )

        # shape (B, T)
        pos_logits = torch.stack(logits_list, dim=0)
        neg_logits = torch.stack(negative_logits_list, dim=0)

        # shape (2, B, T)
        logits = torch.stack([pos_logits, neg_logits], dim=0)
        probs = F.softmax(input=logits, dim=0)

        # shape (B, T)
        preds = probs[0]  # positive predictions

        return self._compute_evaluation_metrics(
            probs=preds.detach().cpu().numpy(),
            labels=np.array(labels_list),
            split=split,
        )

    def iter_dataloaders(self):
        return

    def model_forward(self, batch_input):
        return

    def compute_loss(self, loss_name, loss_fn, batch_input, outputs):
        return