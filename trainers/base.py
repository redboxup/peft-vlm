import gc
import torch
import logging
from abc import ABC, abstractmethod
import numpy as np
import tqdm.auto as tqdm
from typing import Dict, Any, List
from contextlib import nullcontext
from torch.utils.data import DataLoader
from torch.optim import optimizer, lr_scheduler
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from utils import AverageMeter
from configs import ProjectConf
from utils.eval import evaluate, bootstrap
from utils.shuffle import GlobalBatchShuffler


class BaseTrainer(ABC):
    registered_trainers: Dict[str, Any]
    best_val_acc: float = 0

    def __init__(
        self,
        configs: ProjectConf,
        model: torch.nn.Module,
        transforms: Dict[str, Any],
        dataloaders: Dict[str, DataLoader],
        optimizer: optimizer.Optimizer,
        scheduler: lr_scheduler.LRScheduler,
        logger: logging.Logger,
    ):
        self.configs = configs
        self.logger = logger

        self.device = configs.device
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available, Exiting...')

        self.model = model.to(self.device)
        self.transforms = transforms
        self.dataloaders = dataloaders
        self.losses = self.get_losses()

        if configs.trainer.prec == 'amp':
            self.model.float()  # convert model weights to fp32
            self.scaler = GradScaler(device=self.device)

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.start_epoch = 0
        self.verbose(configs.verbose)

    def post_init(self, **kwargs):
        pass

    @abstractmethod
    def get_losses(self):
        raise NotImplementedError()

    def verbose(self, verbose=True):
        self.print = print if verbose else self.noop
        self.tqdm = tqdm.tqdm if verbose else self.noop_iter

    @staticmethod
    def noop(*args, **kwargs):
        pass

    @staticmethod
    def noop_iter(iterable, **kwargs):
        yield from iterable

    def train(self):
        self.loss_weights = {loss: 1.0 for loss in self.losses}
        self.loss_weights.update({
            name: param
            for name, param in self.configs.trainer.items()  # type: ignore
            if name in self.losses
        })

        if self.configs.trainer.move_to_host:
            self.logger.info("Moving batches to host memory...")
            cache_dir = (
                self.configs.rootpath
                / 'data'
                / 'cache'
                / self.configs.dataset.datasetname
            )
            use_cached_dataset = self.configs.trainer.cache_dataset
            is_cache_available = (
                (cache_dir / 'train.pt').exists() and
                (cache_dir / 'val.pt').exists()
            )
            create_cache = use_cached_dataset and not is_cache_available

            if use_cached_dataset and is_cache_available:
                self.logger.info('Loading cache to host-memory')
                self.batches = torch.load(
                    cache_dir / 'train.pt', weights_only=False
                )
                self.val_batches = torch.load(
                    cache_dir / 'val.pt', weights_only=False
                )

            else:
                self.batches = list(
                    self.tqdm(
                        self.iter_dataloaders(),
                        total=len(self.dataloaders["train"]),
                        desc="Moving train split to host memory",
                    )
                )
                self.val_batches = list(
                    self.tqdm(
                        self.dataloaders["val"],
                        desc="Moving val split to host memory",
                    )
                )
                if create_cache:
                    self.logger.info('Saving a cache of the dataset')
                    cache_dir.mkdir(exist_ok=True, parents=True)
                    torch.save(
                        self.batches,
                        str(cache_dir / 'train.pt'),
                        pickle_protocol=4
                    )
                    torch.save(
                        self.val_batches,
                        str(cache_dir / 'val.pt'),
                        pickle_protocol=4,
                    )

        else:
            self.logger.info("Using lazy loading dataloader")
            self.batches = self.iter_dataloaders()
            self.val_batches = None

        for epoch in range(self.start_epoch, self.configs.trainer.epochs):
            self.model.train()
            self.epoch = epoch

            train_losses, extra_data = self.train_epoch()

            if self.scheduler:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            self.logger.info(f'Epoch: {epoch} | LR: {current_lr:.1e}')
            msg = ' '.join(
                [
                    f'{loss_name} = '
                    f'{self.losses[loss_name].__class__.__name__}:{loss:.4f}'
                    for loss_name, loss in train_losses.items()
                ]
            )
            self.logger.info(msg)

            # is_last_epoch = self.epoch == self.configs.trainer.epochs - 1
            # is_save_interval = epoch % self.configs.trainer.save_interval == 0
            #
            # if is_save_interval or is_last_epoch:
            # self.save_checkpoint(epoch)

            if 'best_val' in extra_data:
                self.save_checkpoint(epoch, best=True)

    @abstractmethod
    def iter_dataloaders(self):
        raise NotImplementedError()

    @abstractmethod
    def model_forward(self, batch_input):
        raise NotImplementedError()

    def train_epoch(self):
        avg_losses = {loss_name: AverageMeter() for loss_name in self.losses}

        # use an autocontext if precision is 'amp'
        context = (
            autocast(self.device)
            if self.configs.trainer == 'amp'
            else nullcontext()
        )

        if self.configs.trainer.move_to_host:
            self.batches = GlobalBatchShuffler.shuffle_batches(
                list_batches=self.batches,
                batch_size=self.configs.trainer.batch_size,
                total_len=len(
                    self.dataloaders['train'].dataset  # type: ignore
                ),
                seed=self.epoch
            )

        with context:
            # iterate over dataloader
            for batch_idx, batch_input in (
                enumerate(
                    self.tqdm(
                        self.batches,
                        total=len(self.dataloaders['train']),
                        desc=f"Epoch: {self.epoch}"
                    )
                )
            ):
                _input = batch_input

                batch_size = _input['img'].shape[0]
                if batch_size == 0:
                    continue

                outputs = self.model_forward(batch_input)

                loss_values = {
                    loss_name: self.compute_loss(
                        loss_name, loss, batch_input, outputs
                    )
                    for loss_name, loss in self.losses.items()
                }

                loss = torch.stack(
                    [
                        self.loss_weights[loss] * loss_values[loss]
                        for loss in loss_values
                    ]
                ).sum()

                self.optimizer.zero_grad()

                if self.configs.trainer.prec == 'amp':
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                for loss_name, loss_value in loss_values.items():
                    avg_losses[loss_name].update(
                        loss_value.item(),
                        batch_size
                    )

                # # In case of OOMs
                # if batch_idx % 10 == 0:
                #     gc.collect()
                #     torch.cuda.empty_cache()
                #     torch.cuda.synchronize()

        avg_losses = {
            loss_name: loss.avg
            for loss_name, loss in avg_losses.items()
        }

        extra_data = self.post_train_epoch()
        return avg_losses, extra_data

    def post_train_epoch(self):
        results_df = self.evaluate(split='val')

        _mean = results_df.mean(axis=1).iloc[0]
        self.logger.info(f'Val Split: {_mean}')

        results_dict = {'mAUC': _mean}
        if _mean > self.best_val_acc:
            self.best_val_acc = max(
                self.best_val_acc,
                _mean
            )
            results_dict['best_val'] = _mean

        return results_dict

    @abstractmethod
    def compute_loss(self, loss_name, loss_fn, batch_input, outputs):
        raise NotImplementedError()

    def post_load_checkpoint(self, checkpoint_data):
        pass

    def save_checkpoint(
        self, epoch: int, best=False, suffix='', **extra_kwargs
    ):
        self.logger.info('Saving state dictionary to checkpoint')

        if self.scheduler:
            extra_kwargs['scheduler'] = self.scheduler.state_dict()

        if self.optimizer:
            extra_kwargs['optimizer'] = self.optimizer.state_dict()

        checkpoint_data = {
            'model': self.model.state_dict(),
            'epoch': epoch,
            **extra_kwargs
        }

        save_dir = self.configs.experiment_dir / 'weights'
        if best:
            torch.save(checkpoint_data, save_dir / f"best_epoch{suffix}.pth")
        else:
            torch.save(
                checkpoint_data,
                save_dir / f"epoch_{epoch}{suffix}.pth"
            )

    def load_checkpoint(
        self, epoch: int = 0, best: bool = False, suffix: str = ''
    ) -> None:
        weights_name = (
            "best_epoch.pth" if best
            else f"epoch_{epoch}{suffix}.pth"
        )
        state_dict_path = (
            self.configs.experiment_dir /
            'weights' /
            weights_name
        )
        state_dict: Dict[str, Any] = torch.load(
            state_dict_path,
            weights_only=False
        )
        for key, state in state_dict.items():
            module = getattr(self, key, None)

            if hasattr(module, 'load_state_dict'):
                module.load_state_dict(state)  # type: ignore
                self.logger.info(
                    f"{key} module has been loaded successfully"
                )
            else:
                setattr(self, key, state)
                self.logger.info(
                    f"{key} module has been set as trainer attribute"
                )

        self.logger.info(
            "[green]Loaded state dictionary successfully![/green]"
        )

    def evaluate(self, split: str) -> Any:
        return

    def _compute_evaluation_metrics(
            self, probs: np.ndarray, labels: np.ndarray, split: str):
        """Compute evaluation metrics and save results if test split."""
        self.logger.info("[green]Computing evaluation metrics...[/green]")
        cxr_results = evaluate(
            y_pred=probs,
            y_true=labels,
            cxr_labels=self.configs.dataset.cxr_labels
        )

        if split == 'test':
            cxr_results_ = cxr_results.T.reset_index()
            cxr_results_.columns = ['label', 'auc']
            cxr_results_['label'] = (
                cxr_results_['label']
                .str
                .replace('_auc$', '', regex=True)
            )
            cxr_results_.to_csv(
                self.configs.trainer.results_dir / 'ID_test_auc.csv',
                index=False
            )
            # bootstrap_results = bootstrap(
            #     y_pred=probs,
            #     y_true=labels,
            #     cxr_labels=self.configs.dataset.cxr_labels,
            #     verbose=self.configs.verbose,
            # )
            # bootstrap_results[1].to_csv(
            #     (
            #          self.configs.trainer.results_dir /
            #          'ID_ConfInterval_auc.csv'
            #     ),
            #     index=False
            # )

        return cxr_results

    def forward_pass(self, split: str) -> Any:
        return

    @classmethod
    def register_trainer(cls, trainer_name: str):
        """Decorator to register custom Trainers

        Args:
            trainer_name (str): Name of the trainer
        """
        def decorator(trainer_cls):
            cls.registered_trainers[trainer_name] = trainer_cls
            return trainer_cls
        return decorator

    def freeze_model(self):
        """Override this in the child class to freeze parts of the model"""
        pass

    def check_for_nan_params(self):
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"[NaN Detected] Parameter `{name}` contains NaNs")
                return True
        return False

    def check_gradients(self):
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    print(f"NaNs in gradient of: {n}")
                if torch.isinf(p.grad).any():
                    print(f"Infs in gradient of: {n}")


# Registry for trainers
BaseTrainer.registered_trainers = {}
