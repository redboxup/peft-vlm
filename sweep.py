import os
import ray
import json
import copy
import torch
import hydra
import dotenv
import hashlib
import pandas as pd
from pathlib import Path
from ray import tune, train
from filelock import FileLock
from omegaconf import omegaconf
from typing import Dict, Any, Callable
from ray.tune.search.optuna import OptunaSearch

from configs import ProjectConf
from trainers import BaseTrainer
from allconfigs import CONFIG_REGISTRY
from utils.logger import setup_logging

from utils.make import (
    make_datasets, make_dataloaders, make_transforms,
    make_model, make_optimizer, make_scheduler
)


def config_to_hash(configs: ProjectConf, length=8) -> str:
    json_str = json.dumps(configs, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()[:length]


def create_param_space(configs: ProjectConf) -> Dict[str, Any]:
    param_space = {}

    def flatten(configs: omegaconf.DictConfig, prefix=""):
        for key, value in configs.items():
            full_key = f"{prefix}.{key}" if prefix else key

            # if value in omegaconf Dict
            if isinstance(value, omegaconf.DictConfig):
                flatten(value, prefix=full_key)  # type: ignore

            # if value is omegaconf List
            if isinstance(value, omegaconf.ListConfig):
                if all(isinstance(v, bool) for v in value):
                    param_space[full_key] = tune.choice(list(value))
                elif all(isinstance(v, int) for v in value):
                    param_space[full_key] = tune.choice(list(value))
                elif all(isinstance(v, float) for v in value):
                    low, high = min(value), max(value)
                    if low > 0 and high / low >= 100:
                        param_space[full_key] = tune.loguniform(low, high)
                    else:
                        param_space[full_key] = tune.choice(list(value))
                else:
                    param_space[full_key] = tune.choice(list(value))

    flatten(configs.tune)  # type: ignore
    return param_space


def overide_configs(
    ray_configs: Dict[str, Any],
    configs: ProjectConf
) -> ProjectConf:

    overridden_configs = copy.deepcopy(configs)

    for flat_key, value in ray_configs.items():

        if '.' in flat_key:
            # split key into parts
            keys = flat_key.split('.')
            sub_config = overridden_configs
            for k in keys[:-1]:
                sub_config = sub_config[k]  # type: ignore
            sub_config[keys[-1]] = value  # type: ignore

        else:
            if (
                flat_key in CONFIG_REGISTRY
                and value in CONFIG_REGISTRY[flat_key]
            ):
                setattr(
                    overridden_configs,
                    flat_key, CONFIG_REGISTRY[flat_key][value]()
                )
            else:
                raise ValueError(
                    f"Invalid registry override: '{flat_key}={value}' — "
                    f"Available options: "
                    f"{list(CONFIG_REGISTRY.get(flat_key, {}).keys())}"
                )
    return overridden_configs


def trainable_factory(
    global_configs: ProjectConf
) -> Callable[[Dict[str, Any]], None]:
    """Factory to create a Ray Tune trainable function

    Args:
        global_configs (ExpConf): Base Hydra experiment config

    Returns:
        Callable: A function to be used as Ray Tune's trainable,
        accepting Ray's config and returning nothing.
    """
    def trainable(ray_configs):
        local_configs = overide_configs(ray_configs, global_configs)

        # generate experiment directory
        exp_hash = config_to_hash(ray_configs)
        exp_id = f"exp_{exp_hash}"
        trainer_name = (
            f'{local_configs.trainer.name}_{local_configs.trainer.few_shots}'
            if local_configs.trainer.few_shot_sampler
            else f'{local_configs.trainer.name}'
        )
        experiment_dir = (
            local_configs.rootpath /
            'experiments_new' /
            local_configs.dataset.datasetname /
            trainer_name /
            local_configs.trainer.few_shot_sampler /
            exp_id
        )

        logs_dir = experiment_dir / 'logs'
        results_dir = experiment_dir / 'results'
        weights_dir = experiment_dir / 'weights'
        logs_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        weights_dir.mkdir(parents=True, exist_ok=True)
        local_configs.trainer.results_dir = results_dir
        local_configs.experiment_dir = experiment_dir

        logger = setup_logging(logs_dir, verbose=local_configs.verbose)
        logger.info("Logging initialised")

        transforms = make_transforms(
            local_configs, logger
        )
        datasets = make_datasets(
            local_configs, transforms, logger
        )
        dataloaders = make_dataloaders(
            local_configs, datasets, logger
        )
        model_path = str(
            local_configs.rootpath /
            local_configs.weights_dir /
            local_configs.trainer.model_weights
        )
        model = make_model(
            local_configs, model_path, logger
        )
        optimizer = make_optimizer(
            local_configs, model, logger
        )
        if optimizer:
            scheduler = make_scheduler(
                local_configs, optimizer, logger
            )
        else:
            scheduler = None

        logger.info(
            f'Initializing {local_configs.trainer.name} Trainer...'
        )
        trainer: BaseTrainer = BaseTrainer.registered_trainers[
            local_configs.trainer.name
        ](
            local_configs,
            model,
            transforms,
            dataloaders,
            optimizer,
            scheduler,
            logger
        )

        original_post_train_epoch = trainer.post_train_epoch

        def new_post_train_epoch(*args, **kwargs):
            results_df = original_post_train_epoch(*args, **kwargs)
            train.report(results_df)
            return results_df

        trainer.post_train_epoch = new_post_train_epoch

        logger.info("Starting training")
        trainer.train()

        # load the best validation accuracy model weights
        logger.info(
            'Loading weights corresponding to best validation metric'
        )
        state_dict = torch.load(
            weights_dir / 'best_epoch.pth',
            weights_only=False
        )
        trainer.model.load_state_dict(state_dict['model'])
        best_val_epoch = state_dict['epoch']

        logger.info("Evaluating model...")
        # test_df = trainer.evaluate(split='test')
        # avg_test_auc = test_df.mean(axis=1).iloc[0]

        summary_row = {
            'exp_id': exp_id,
            'parameter_space': "-".join(
                f"{k}={v}" for k, v in ray_configs.items()
            ),
            'best_val_epoch': best_val_epoch,
            f'best_val_{local_configs.trainer.metric}': trainer.best_val_acc,
            # f'test_{local_configs.trainer.metric}': avg_test_auc,
        }
        global_csv_path = (
            experiment_dir.parent /
            'train.summary.csv'
        )
        lock_path = str(
            global_csv_path.parent /
            ".train.summary.csv.lock"
        )

        with FileLock(lock_path):
            if global_csv_path.exists():
                df = pd.read_csv(str(global_csv_path))
            else:
                df = pd.DataFrame()
            df = pd.concat(
                [df, pd.DataFrame([summary_row])],
                ignore_index=True
            )
            df.to_csv(str(global_csv_path), index=False)
    return trainable


@hydra.main(config_name='projectconf', version_base=None)
def main(configs: ProjectConf):
    dotenv.load_dotenv('.env')
    configs.rootpath = Path(os.environ['DATAPATH'])

    param_space = create_param_space(configs)
    trainable = trainable_factory(global_configs=configs)

    # ray.init(log_to_driver=False)
    trainer_name = (
        f'{configs.trainer.name}_{configs.trainer.few_shots}'
        if configs.trainer.few_shot_sampler
        else f'{configs.trainer.name}'
    )
    tune_dir = (
        configs.rootpath /
        'experiments_new' /
        configs.dataset.datasetname /
        trainer_name /
        configs.trainer.few_shot_sampler /
        'raytune'
    )

    # resume previous sweep
    if tune.Tuner.can_restore(tune_dir):
        tuner = tune.Tuner.restore(
            path=str(tune_dir),
            trainable=tune.with_resources(
                tune.with_parameters(trainable),
                resources={
                    'cpu': configs.tune.cpus_per_trial,
                    'gpu': configs.tune.gpus_per_trial
                }
            ),
            resume_unfinished=True,
            resume_errored=False,
            restart_errored=False,
        )

    # start new sweep
    else:
        tuner = tune.Tuner(
            trainable=tune.with_resources(
                tune.with_parameters(trainable),
                resources={'cpu': configs.tune.cpus_per_trial,
                           'gpu': configs.tune.gpus_per_trial}
            ),
            tune_config=tune.TuneConfig(
                metric=configs.trainer.metric,
                num_samples=configs.tune.num_samples,
                mode='max',
                search_alg=OptunaSearch(),
            ),
            param_space=param_space,
            run_config=train.RunConfig(
                name='raytune',
                storage_path=str(tune_dir),
                log_to_file=False,
            ),
        )

    results = tuner.fit()
    best_run = results.get_best_result()

    print("Search Ended")
    print("> Config:", best_run.config)
    print("> Metrics:", best_run.metrics)
    ray.shutdown()


if __name__ == '__main__':
    main()
