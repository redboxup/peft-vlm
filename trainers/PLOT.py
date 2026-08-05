import torch
import numpy as np
from logging import Logger
from rich.console import Console
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Any
from torch.optim import optimizer, lr_scheduler

from configs import ProjectConf
from models.PLOT import PLOTCLIP
from trainers import BaseTrainer

@BaseTrainer.register_trainer('plot')
class PLOTTrainer(BaseTrainer):
    def _init__ (
        self,
        configs: ProjectConf,
        model: PLOTCLIP,
        dataloaders: Dict[str, DataLoader],
        optimizer: optimizer.Optimizer,
        scheduler: lr_scheduler.LRScheduler,
        logger: Logger
    ):
        super().__init__(
            configs, model, dataloaders, optimizer, scheduler, logger
        )
        self.model: PLOTCLIP
        self.dataloaders: Dict[str, DataLoader]
        self.console = Console()
        self.logger = logger
        self.freeze_model()

    def freeze_model(self):
        for name, param in self.model.named.parameters():
            param.requires_grad = False

        for name, param in self.model.prompt_learner.named_parameters():
            param.requires_grad = True
            self.logger.info(f"[{name}] layer unfrozen")

    def get_losses(self) -> Dict[str, Any]:
        return dict(
            ce = torch.nn.CrossEntropyLoss()
        )

    def iter_dataloaders(self):
        return self.datalaoders['train']

    def model_forward(self, batch_input):
        images = batch_input['img']
        images = images.to(self.device)
        # output: [batch_size, num_classes]
        output = self.model(images)
        return output
    
    def compute_loss(self, loss_name, loss_fn, batch_input, outputs):
        labels = batch_input['labels'].to(self.device)
        mulitclass_labels = labels.argmax(dim=-1)
        outputs = torch.nn.functional.softmax(outputs, dim=-1)
        # print(f"")

        if loss_name == 'ce':
            return loss_fn(input=outputs, target=mulitclass_labels)

        
    def forward_pass(self, split: str) -> Tuple[List[Any], List[Any]]:
        """
        Forward pass for the model.
        Args:
            split (str): The split of the dataset (e.g., 'train', 'val', 'test').
        Returns:
            Tuple[List[Any], List[Any]]: A tuple containing the outputs and labels.
        """
        logits_list= []
        labels_list = []

        with torch.no_grad():
            for data in self.tqdm(
                self.dataloaders[split],
                desc=f"[{split}] Forward Pass"
            ):
                images, labels = data['img'].to(self.device), data['labels']
                logits = self.model(images)
                
                logits_list.extend(logits.cpu())
                labels_list.extend(labels.numpy())
            
        return logits_list, labels_list
    
    def evaluate(self, split: str) -> Dict[str, Any]:
        self.logger.info(f"[bold blue] Evaluate on {split} data")
        self.model.eval()

        logits_list, labels_list = self.forward_pass(split=split)

        # *_logits: [num_samples, num_classes]
        logits = torch.stack(logits_list)
        labels = torch.tensor(labels_list)

        return self._compute_evaluation_metrics(
            probs=logits.detach().cpu().numpy(),
            labels=np.array(labels),
            split=split
        )