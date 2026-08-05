import torch
import numpy as np
from logging import Logger
from rich.console import Console
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Any
from torch.optim import optimizer, lr_scheduler

from configs import ProjectConf
from models.CoOp import CoOpCLIP
from trainers import BaseTrainer


@BaseTrainer.register_trainer('coop')
class CoOpTrainer(BaseTrainer):
    def __init__(
        self,
        configs: ProjectConf,
        model: CoOpCLIP,
        transforms: Dict[str, Any],
        dataloaders: Dict[str, DataLoader],
        optimizer: optimizer.Optimizer,
        scheduler: lr_scheduler.LRScheduler,
        logger: Logger
    ):
        super().__init__(
            configs,
            model,
            transforms,
            dataloaders,
            optimizer,
            scheduler,
            logger,
        )
        self.model: CoOpCLIP
        self.transforms: Dict[str, Any]
        self.dataloaders: Dict[str, DataLoader]
        self.console = Console()
        self.logger = logger
        self.freeze_model()

    def freeze_model(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        for name, param in self.model.prompt_learner.named_parameters():
            param.requires_grad = True
            self.logger.info(f'[{name}] layer unfrozen')

    def get_losses(self) -> Dict[str, Any]:
        return dict(
            ce=torch.nn.CrossEntropyLoss(),
        )

    def iter_dataloaders(self):
        return self.dataloaders['train']

    def model_forward(
        self, batch_input: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
        images = batch_input['img']
        images = self.transforms['train'](images.to(self.device))
        output = self.model(images)
        return output

    def compute_loss(self, loss_name, loss_fn, batch_input, outputs):
        labels = batch_input['labels'].to(self.device)
        # print(f"Labels shape: {labels.shape}")  # Debugging line
        multiclass_labels = labels.argmax(dim=1)
        # print(f"Multiclass labels shape: {multiclass_labels.shape}")
        # print(f"outputs shape: {outputs.shape}")  # Debugging line
        # import sys
        # sys.exit(0)  # Stop execution for debugging

        if loss_name == 'ce':
            return loss_fn(input=outputs, target=multiclass_labels)

    def forward_pass(
        self,
        split: str
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[np.ndarray]]:
        positive_logits_list, negative_logits_list, labels_list = [], [], []

        if split == 'val' and self.configs.trainer.move_to_host:
            iterable = self.val_batches
        else:
            iterable = self.dataloaders[split]

        with torch.inference_mode():
            for data in self.tqdm(
                iterable,
                desc=f'[{split}] Forward Pass'
            ):
                images = self.transforms[split](
                    data['img'].to(self.device)
                    )
                labels = data['labels']

                # logits: [batch, 2, num_classes]
                logits = self.model.forward_with_neg(images)
                neg_logits = logits[:, 0, :]
                pos_logits = logits[:, 1, :]

                positive_logits_list.extend(pos_logits.cpu())
                negative_logits_list.extend(neg_logits.cpu())
                labels_list.extend(labels.numpy())

        return positive_logits_list, negative_logits_list, labels_list

    def evaluate(self, split: str):
        self.logger.info(f"[bold blue] Evaluate on {split} data")
        self.model.eval()

        pos_logits_list, neg_logits_list, labels_list = (
            self.forward_pass(split=split)
        )

        # *_logits: [num_samples, num_classes]
        pos_logits = torch.stack(pos_logits_list, dim=0)
        neg_logits = torch.stack(neg_logits_list, dim=0)

        # logits: [num_samples, num_classes, 2]
        logits = torch.stack([neg_logits, pos_logits], dim=2)

        # logits: [num_samples * num_classes, 2]
        logits = logits.view(-1, 2)

        # probs: [num_samples * num_classes, 2]
        probs = torch.softmax(input=logits, dim=-1)

        # probs: [num_samples, num_classes]
        probs = probs[:, 1].view(-1, len(self.configs.dataset.cxr_labels))


        return self._compute_evaluation_metrics(
            probs=probs.detach().cpu().numpy(),
            labels=np.array(labels_list),
            split=split
        )
