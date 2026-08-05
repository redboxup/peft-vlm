import torch
import logging
from pathlib import Path
import kornia.augmentation as K
from torch.utils.data import DataLoader
from typing import Dict, Union, Optional, Any
from torch.optim import lr_scheduler, optimizer
from kornia.augmentation import AugmentationSequential
from torchvision.transforms import Compose, transforms

from clip.clip import load
from models.base import BaseCLIP
from configs import ProjectConf
from utils.datasets import CXRDatasets, DictImageFolder
from utils.sampler import (
    NShotRandomSampler,
    NShotStartifiedSampler,
    NShotMultiClassSampler,
    VisionSampler,
)


def make_model(
    configs: ProjectConf,
    model_path: Union[str, Path],
    logger: logging.Logger
) -> BaseCLIP:
    """make function that will return the model required for the trainer

    Args:
        configs (ProjectConf): Configuration file for the experiment
        model_path (Union[str, Path]): load CLIP model from existing 
                                        weights

    Returns:
        BaseCLIP: CLIP model inheriting BaseCLIP
    """
    device = torch.device(configs.device)

    # load clip pre-trained model
    if configs.trainer.pretrained:
        logger.info('Model initialised from pretrained openai CLIP')
        model, _ = load(configs.model, device=device, jit=False, model_i=configs.trainer.name)
    else:
        raise ValueError('Training with random initialisation not supported')

    # if a model_path is provided, load in weights to backbone
    if model_path is not None:
        logger.info(f'Loading weights from {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=device,
                                         weights_only=True))
        model.float()

    model = BaseCLIP.registered_models[configs.trainer.name](configs, model)
    return model


def make_transforms(
    configs: ProjectConf, logger: logging.Logger,
) -> Dict[str, Any]:
    imsize = configs.trainer.img_size
    transforms_dict = {}
    transforms_dict['to_tensor'] = transforms.ToTensor()
    transforms_dict['resize_and_normalise_k'] = AugmentationSequential(
        K.Resize((imsize, imsize)),
        K.Normalize(
            mean=torch.tensor([0.5, 0.5, 0.5]),
            std=torch.tensor([0.5, 0.5, 0.5]),
        )
    )
    transforms_dict['resize_and_normalise'] = Compose([
        transforms.ToTensor(),
        transforms.Resize(
            size=(configs.trainer.img_size, configs.trainer.img_size)
        ),
        transforms.Normalize(mean=0.5, std=0.5),
    ])
    if configs.trainer.transforms == 'strong':
        logger.info('Using Strong Transforms')
        transforms_dict['train'] = AugmentationSequential(
            K.RandomResizedCrop(size=(imsize, imsize)),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomGrayscale(p=0.2),
            K.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.2,
            ),
        )
    elif configs.trainer.transforms == 'weak':
        logger.info('Using Weak transforms')
        transforms_dict['train'] = AugmentationSequential(
            K.RandomResizedCrop(size=(imsize, imsize)),
            K.RandomHorizontalFlip(p=0.5),
        )
    elif configs.trainer.transforms == 'randaugment':
        logger.info('Using RandomAugmentations as Transforms')
        transforms_dict['train'] = K.auto.RandAugment(n=2, m=9)
    else:
        raise ValueError(
            f"{configs.trainer.transforms} augmentations is not supported"
        )
    transforms_dict['val'] = K.RandomHorizontalFlip(p=0.0)
    transforms_dict['test'] = transforms_dict['resize_and_normalise_k']
    return transforms_dict


def make_datasets(
    configs: ProjectConf,
    transforms: Dict[str, Compose],
    logger: logging.Logger,
) -> Dict[str, CXRDatasets]:
    """creates ID and OOD datasets

    Args:
        configs (ProjectConf): Configurations file for the experiment
        transforms (Dict[str, Compose]): Dictionary of transforms

    Returns:
        Dict[str, CXRDatasets]: Dictionary of datasets
    """
    datasets = {}
    for name in ['train', 'val', 'test']:
        imagedir_fpath = getattr(configs.dataset, f'{name}_images_filepath')
        labels_fpath = getattr(configs.dataset, f'{name}_labels_filepath')
        transforms_ = (
            transforms['to_tensor'] if name == 'test'
            else transforms['resize_and_normalise']
        )

        datasets.update({
            name: CXRDatasets(
                dataset_name=configs.dataset.datasetname,
                imagedir_filepath=configs.rootpath / imagedir_fpath,
                labels_filepath=configs.rootpath / labels_fpath,
                cxr_labels=configs.dataset.cxr_labels,
                transforms=transforms_,
            )})

    for name in ['irma', 'mura', 'boneage']:
        datasets.update({
            name: DictImageFolder(
                root=configs.rootpath / configs.ood.dataset_dict[name],
                transform=transforms['to_tensor']),
        })
    return datasets


def make_dataloaders(
    configs: ProjectConf,
    datasets: Dict[str, CXRDatasets],
    logger: logging.Logger,
) -> Dict[str, DataLoader]:
    """Creates dataloaders for the datasets
    Args:
        configs (ProjectConf): Configurations file for the experiment
        datasets (Dict[str, CXRDatasets]): Dictionary of datasets

    Returns:
        Dict[str, DataLoader]: Dictionary of dataloaders
    """
    dataloaders = {}
    for name in ['train', 'val', 'test', 'irma', 'mura', 'boneage']:

        is_train = name == 'train'
        batch_size = (
            configs.trainer.batch_size if is_train
            else configs.eval.batch_size
        )
        num_workers = (
            configs.trainer.num_workers if is_train
            else configs.eval.num_workers
        )

        sampler = None
        if is_train:
            sampler_map = {
                'random': NShotRandomSampler,
                'multiclass': NShotMultiClassSampler,
                'stratified': NShotStartifiedSampler,
                'vision_classifier': VisionSampler,
            }
            sampler_type = configs.trainer.few_shot_sampler
            Sampler = sampler_map.get(sampler_type)

            if Sampler is None:
                raise ValueError(
                    f'{configs.trainer.sampler} is '
                    'not a valid sampler'
                )

            logger.info(
                f"Using sampler {sampler_type},"
            )
            sampler = Sampler(
                dataset=datasets[name],
                n_shot=configs.trainer.few_shots,
            )

        dataloaders.update({
            name: DataLoader(
                datasets[name],
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
            )
        })

    return dataloaders


def make_optimizer(
    configs: ProjectConf,
    model: BaseCLIP,
    logger: logging.Logger
) -> Optional[optimizer.Optimizer]:
    """make function to create an optimizer given the configurations

    Args:
        configs (ProjectConf): Experiment configurations files
        model (BaseCLIP): model to be optimized over

    Returns:
        torch.optim.Optimizer: Optimizer object (e.g., Adam, sgd)
    """
    # get optimizer configurations
    optim_configs = {
        k: v for k, v in configs.optimizer.items()  # type: ignore
        if k != 'name'
    }

    if model.optim_params is not None:
        # instantiate the optimizer
        if configs.optimizer.name == 'adam':
            logger.info('Using Adam optimizer')
            return torch.optim.Adam(  # type: ignore
                model.optim_params,
                configs.trainer.learning_rate,
                **optim_configs
            )
        elif configs.optimizer.name == 'sgd':
            logger.info('Using SGD optimizer')
            return torch.optim.SGD(  # type: ignore
                model.optim_params,
                configs.trainer.learning_rate,
                **optim_configs
            )
    else:
        logger.info('Using NO optimizer')
        return None


def make_scheduler(
    configs: ProjectConf,
    optimizer: optimizer.Optimizer,
    logger: logging.Logger
) -> Union[lr_scheduler.LRScheduler, lr_scheduler.SequentialLR]:
    """make function to create a learning rate scheduler

    Args:
        configs (ExpConf): Configurations file for experiment
        optimizer (torch.optim.Optimizer): Optimizer object

    Raises:
        ValueError: Only that scheduler mentioned is not supported

    Returns:
        LRScheduler: learning rate scheduler for training
    """
    sched_configs = {
        k: v for k, v in configs.scheduler.items()  # type: ignore
        if k != 'name'
    }

    if configs.scheduler.name == 'linear':
        logger.info("Using Linear LR Scheduler")
        main_scheduler = lr_scheduler.LinearLR(
            optimizer, **sched_configs
        )
    elif configs.scheduler.name == 'cosine':
        logger.info("Using Cosine LR Scheduler")
        main_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, **sched_configs
        )
    elif configs.scheduler.name == 'none':
        logger.info("Using No LR Scheduler")
        main_scheduler = lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=1
        )
    else:
        raise ValueError(f'Unsupported sceduler: {configs.scheduler.name}')

    if configs.warmup.use_warmup:
        warmup_scheduler = lr_scheduler.LinearLR(
            optimizer, start_factor=configs.warmup.start_factor,
            end_factor=1.0, total_iters=configs.warmup.epochs,
        )

        # combine warmup_scheduler & main_scheduler
        logger.info('Using Warmup in learning rate')
        scheduler = lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler],
            milestones=[configs.warmup.epochs],
        )
        return scheduler
    return main_scheduler
