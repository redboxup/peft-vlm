import torch
import numpy as np
from logging import Logger
from rich.console import Console
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Any
from torch.optim import optimizer, lr_scheduler
from losses.GalLoP import GlobalLocalLoss, topk_reduce

from configs import ProjectConf
from models.GalLoP import GalLoPCLIP
from trainers import BaseTrainer


@BaseTrainer.register_trainer('gallop')
class GalLoPTrainer(BaseTrainer):
    def __init__(
        self,
        configs: ProjectConf,
        model: GalLoPCLIP,
        transforms: Dict[str, Any],
        dataloaders: Dict[str, DataLoader],
        optimizer: optimizer.Optimizer,
        scheduler: lr_scheduler.LRScheduler,
        logger: Logger
    ):
        self.topk = [configs.trainer.batch_size*i for i in range(1, configs.trainer.n_local_prompts + 1)]
        super().__init__(
            configs, model,transforms, dataloaders, optimizer, scheduler, logger
        )
        self.model: GalLoPCLIP
        transforms: Dict[str, Any]
        self.dataloaders: Dict[str, DataLoader]
        self.console = Console()
        self.logger = logger
        self.freeze_model()

    def freeze_model(self):
        # print(f"learning parametrs: {self.model.trainable_params}") 
        for name, param in self.model.named_parameters():
            if name in self.model.trainable_params:
                param.requires_grad = True
                # self.logger.info(f"Training parameter: {name}")
            else:
                param.requires_grad = False
                # self.logger.info(f"Freezing parameter: {name}")
            # print(f"Parameter: {name}, requires_grad: {param.requires_grad}")


    def get_losses(self) -> Dict[str, Any]:
        return dict(
            gl=GlobalLocalLoss(
                use_global_loss=self.configs.trainer.use_global_loss,
                use_local_loss=self.configs.trainer.use_local_loss,
                topk=self.topk,)
        )

    def iter_dataloaders(self):
        return self.dataloaders['train']

    def model_forward(self, batch_input):
        images = batch_input['img']
        images = images.to(self.device)
        # output: [batch_size, num_classes]
        outputs = self.model(images)
        return outputs

    def compute_loss(self, loss_name, loss_fn: GlobalLocalLoss, batch_input, outputs):
        labels = batch_input['labels'].to(self.device)
        multiclass_labels = labels.argmax(dim=1)

        if loss_name == 'gl':
            return loss_fn(global_logits=outputs[0],
                            local_logits=outputs[1],
                            targets=multiclass_labels,
                            logit_scale=self.model.logit_scale.exp())

    def forward_pass(
        self,
        split: str
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[np.ndarray]]:
        global_logits_lst, local_logits_lst, labels_list = [], [], []

        if split == 'val' and self.configs.trainer.move_to_host:
            iterable = self.val_batches
        else:
            iterable = self.dataloaders[split]

        with torch.inference_mode():
            for data in self.tqdm(
                iterable,
                desc=f'[{split}] Forward Pass'
            ):
                images, labels = data['img'].to(self.device), data['labels']

                global_logits, local_logits = self.model(images)
                global_logits_lst.extend(global_logits.cpu())
                local_logits_lst.extend(local_logits.cpu())
                labels_list.extend(labels.numpy())
        # global_logits_lst: List[Tensor]. List len = n_samples
        # Tensor shape: [n_cls, n_global_prompts]
        # local_logits_lst: List[Tensor]. List len = n_samples
        # Tensor shape: [grid**2, n_cls, local_prompts]

        return global_logits_lst, local_logits_lst, labels_list

    def evaluate(self, split: str):
        self.logger.info(f"[bold blue] Evaluate on {split} data")
        self.model.eval()

        global_logits_lst, local_logits_lst, labels_list = (
            self.forward_pass(split=split)
        )

        global_logits = torch.stack(global_logits_lst, dim=0)
        local_logits = torch.stack(local_logits_lst, dim=0)
        # global_logits: [num_samples, n_cls, prompts_batch_size]
        # local_logits: [num_samples, grid**2, n_cls, n_local_prompts]
        local_logits = topk_reduce(
            local_logits=local_logits,
            topk=self.topk
        )
        # local_logits: [num_samples, n_cls, n_prompts]

        global_logits = global_logits.mean(dim=-1)
        local_logits = local_logits.mean(dim=-1)
        # [num_samples, n_cls]

        preds = global_logits + local_logits
        # [num_samples, n_cls]
        
        return self._compute_evaluation_metrics(
            probs=preds.detach().cpu().numpy(),
            labels=np.array(labels_list),
            split=split
        )