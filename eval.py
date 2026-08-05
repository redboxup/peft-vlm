import os
import hydra
import dotenv
import pandas as pd
from pathlib import Path
from rich.table import Table
from rich.panel import Panel
from rich.console import Console

from configs import ProjectConf
from sweep import overide_configs
from trainers.base import BaseTrainer
from detectors import BaseOODDetector
from utils.logger import setup_logging

from utils.make import (
    make_transforms, make_datasets, make_dataloaders, make_model,
    make_optimizer, make_scheduler
)


def parse_string_to_dict(string: str):
    def convert(value):
        if value == 'True':
            return True
        if value == 'False':
            return False
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                continue
        return value
    hyperparameter_dict = {}
    for item in string.split('-'):
        if len(item.split('=')) == 2:
            for key, value in [item.split('=')]:
                if key == "trainer.learning_rate":
                    value = convert(value[:-1])
                    # 62.90204602046142e type str --> 62.90204602046142 type float
                    if value > 0.1:
                        value = value * 1e-05  # Adjusting large learning rates
                    else:
                        value = value
                hyperparameter_dict[key] = convert(value)
    return hyperparameter_dict

@hydra.main(config_name='projectconf', version_base='1.3')
def main(configs: ProjectConf):
    console = Console()
    dotenv.load_dotenv('.env')
    configs.rootpath = Path(os.environ['DATAPATH'])

    trainer_name = (
        f'{configs.trainer.name}_{configs.trainer.few_shots}'
        if configs.trainer.few_shot_sampler
        else f'{configs.trainer.name}'
    )
    eval_dir = (
        configs.rootpath /
        'experiments_new' /
        configs.dataset.datasetname /
        trainer_name /
        configs.trainer.few_shot_sampler
    )

    summary_path = eval_dir / 'train.summary.csv'
    # check if summary file exists
    if not summary_path.exists():
        raise FileNotFoundError(f'Summary file not found: {summary_path}')  

    summary_df = (
        pd.read_csv(summary_path)
        .rename(columns=lambda x: x.strip() if isinstance(x, str) else x)
        .apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
        .sort_values(f'best_val_{configs.trainer.metric}', ascending=False)
        .iloc[:configs.eval.top_k]
    )
    
    for idx, row in summary_df.iterrows():
        hyperparam_dict = parse_string_to_dict(row['parameter_space'])
        table = Table(
            title='Hyperparameters',
            show_header=True,
            header_style='bold cyan'
        )
        table.add_column('Name')
        table.add_column('Value')
        for k, v in hyperparam_dict.items():
            table.add_row(k, str(v))

        # Update configs with run-specific parameters
        local_configs = overide_configs(
            hyperparam_dict,
            configs=configs
        )

        local_configs.experiment_dir = (
            configs.rootpath /
            'experiments_new' /
            configs.dataset.datasetname /
            trainer_name /
            configs.trainer.few_shot_sampler /
            f"{row['exp_id']}"
        )

        logger = setup_logging(
            local_configs.experiment_dir / 'logs',
            name='evaluation',
            verbose=False
        )
        logger.info("Logging initalised")

        local_configs.trainer.results_dir = (
            local_configs.experiment_dir /
            'results'
        )

        transforms = make_transforms(local_configs, logger)
        datasets = make_datasets(local_configs, transforms, logger)
        dataloaders = make_dataloaders(local_configs, datasets, logger)
        model_path = str(
            configs.rootpath /
            configs.weights_dir /
            configs.trainer.model_weights
        )
        model = make_model(local_configs, model_path, logger)
        optimizer = make_optimizer(local_configs, model, logger)
        if optimizer:
            scheduler = make_scheduler(local_configs, optimizer, logger)
        else:
            scheduler = None

        trainer: BaseTrainer = (
            BaseTrainer.registered_trainers[local_configs.trainer.name](
                local_configs,
                model,
                transforms,
                dataloaders,
                optimizer,
                scheduler,
                logger,
            )
        )
        trainer.load_checkpoint(best=True)

        test_results_df = trainer.evaluate(split='test')
        console.print(
            Panel(
                table,
                title=f"Experiment: [bold magenta]{row['exp_id']}\
                    [/bold magenta]"
            )
        )
        console.print(
            f"[blue] Best Validation Accuracy: [/blue]"
            f"{row[f'best_val_{configs.trainer.metric}']:.4f}"
            f" @ Epoch: {trainer.epoch}"
        )
        console.print(
            f"[blue] Test Accuracy: [/blue]"
            f"{test_results_df.mean(axis=1).iloc[0]:.4f}"
        )

        detector: BaseOODDetector = (
            BaseOODDetector.registered_detectors[configs.eval.name](
                local_configs,
                model,
                transforms,
                dataloaders,
            )
        )

        ood_dict = detector.evaluate()

        results_table = Table.grid(padding=(0, 1))
        results_table.add_column("Metric", style="cyan", justify="left")
        results_table.add_column("Value", style="magenta", justify="right")

        for key, value in ood_dict.items():
            results_table.add_row(key, f"{value:.4f}")

        console.print(
            Panel(
                results_table,
                title="[blue]OOD Evaluation Metrics[/blue]",
                border_style="green",
            )
        )

        for key, value in ood_dict.items():
            summary_df.at[idx, key] = value

        summary_df.to_csv(
            eval_dir / f'{configs.eval.name}.summary.csv',
            index=False
        )


if __name__ == '__main__':
    main()
