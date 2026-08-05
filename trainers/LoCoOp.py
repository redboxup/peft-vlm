import torch
from typing import Dict, Tuple, Callable

from trainers import BaseTrainer
from trainers.CoOp import CoOpTrainer
from losses.LoCoOp import EntropySelectTopK


@BaseTrainer.register_trainer("locoop")
class LoCoOpTrainer(CoOpTrainer):
    def get_losses(self):
        return dict(
            ce=torch.nn.CrossEntropyLoss(),
            en=EntropySelectTopK(
                top_k=self.configs.trainer.top_k,  # type: ignore
                num_of_local_feature=7
            ),
        )

    def model_forward(  # type: ignore
        self, batch_input: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        images = batch_input['img']
        images = images.to(self.device)
        output = self.model.forward_feats(images)

        return output

    def compute_loss(
        self,
        loss_name: str,
        loss_fn: Callable,
        batch_input: Dict[str, torch.Tensor],
        outputs: Tuple[torch.Tensor, torch.Tensor]
    ):
        # labels: [batch_size, num_cls]
        labels = batch_input['labels'].to(self.device)
        # multiclass_labels: [batch_size]
        multiclass_labels = labels.argmax(dim=1)

        if loss_name == 'ce':
            return loss_fn(input=outputs[0], target=multiclass_labels)

        elif loss_name == 'en':
            batch_size, n_cls = outputs[1].shape[0], outputs[1].shape[-1]
            # outputs[1] shape: [b, h, w, n_cls]
            output_local = outputs[1].view(batch_size, -1, n_cls)
            # output_local shape: (b , h*w, n_cls)
            return loss_fn(p=output_local, label=multiclass_labels)
