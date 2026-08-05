import os
import torch
import hydra
import dotenv
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from configs import ProjectConf
from trainers import BaseTrainer
from utils.logger import setup_logging
from utils.make import (
    make_datasets,
    make_dataloaders,
    make_transforms,
    make_model,
    make_optimizer,
    make_scheduler,
)


@hydra.main(config_name='projectconf', version_base=None)
def main(configs: ProjectConf):
    dotenv.load_dotenv('.env')
    configs.rootpath = Path(os.environ['DATAPATH'])

    exp_id = 'exp_' + '0' * 8
    trainer_name = (
        f'{configs.trainer.name}_{configs.trainer.few_shots}'
        if configs.trainer.few_shot_sampler
        else f'{configs.trainer.name}'
    )
    experiment_dir = (
        configs.rootpath /
        'experiments_new' /
        configs.dataset.datasetname /
        trainer_name /
        configs.trainer.few_shot_sampler /
        exp_id
    )
    logs_dir = experiment_dir / 'logs'
    results_dir = experiment_dir / 'results'
    weights_dir = experiment_dir / 'weights'
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    configs.trainer.results_dir = results_dir
    configs.experiment_dir = experiment_dir

    logger = setup_logging(logs_dir)
    logger.info("Logging initalised!")

    logger.info("Preparing dataloaders...")
    transforms = make_transforms(configs, logger)
    datasets = make_datasets(configs, transforms, logger)
    dataloaders = make_dataloaders(configs, datasets, logger)

    # visual verification of dataloaders
    if configs.dataset.verify_dataloaders:
        for name in ['irma', 'mura', 'boneage']:
            print(f"length of {name} dataloader: {len(dataloaders[name])}")
            print(f"batch size: {dataloaders[name].batch_size}")
            data = next(iter(dataloaders[name]))
            grid_images = make_grid(data['img'], normalize=True)
            grid_images = grid_images.permute(1, 2, 0)
            plt.figure(figsize=(15, 15))
            plt.axis('off')
            plt.imshow(grid_images)
            plt.savefig(f"assets/{name}.png")

        # verify few-shot loader is working fine
        if configs.trainer.sampler:
            labels_list = []
            for data in dataloaders['train']:
                labels = data['labels']
                labels_list.extend(labels.argmax(dim=1).numpy())
            assert (
                set(range(len(configs.dataset.cxr_labels)))
                .issubset(labels_list)
            )
            assert (
                len(labels_list) ==
                configs.trainer.few_shots * len(configs.dataset.cxr_labels)
            )

    logger.info("Loading model...")
    model_path = str(
        configs.rootpath /
        configs.weights_dir /
        configs.trainer.model_weights
    )
    model = make_model(configs, model_path, logger)
    model = torch.compile(model)

    optimizer = make_optimizer(configs, model, logger)
    if optimizer:
        scheduler = make_scheduler(
            configs=configs,
            optimizer=optimizer,
            logger=logger
        )
    else:
        scheduler = None

    logger.info("Initializing trainer...")
    trainer: BaseTrainer = (
        BaseTrainer.registered_trainers[configs.trainer.name](
            configs,
            model,
            transforms,
            dataloaders,
            optimizer,
            scheduler,
            logger
        )
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating model...")
    results = trainer.evaluate(split='test')
    print(results)

    print(f"Mean: {results.values.mean()}")


if __name__ == '__main__':
    main()
    logging.shutdown()
