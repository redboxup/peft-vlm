# Architecture

Prompt-learning PEFT methods for CLIP-based medical VLMs, evaluated on chest X-ray datasets. Hydra handles configuration, and Ray Tune handles sweeps.

## Directory map

```
.
├── run.py            Entry point: one experiment (Hydra)
├── sweep.py          Entry point: Ray Tune hyperparameter search
├── eval.py           Entry point: evaluate saved experiments (OOD detection)
├── preprocess.py     Raw dataset -> resized images + split CSVs
├── configs.py        ProjectConf root dataclass + Hydra ConfigStore registrations
├── allconfigs/       Config dataclasses, one file per group
│   ├── train_configs.py      Per-method trainer configs (CoOpConf, GalLoPConf, ...)
│   ├── dataset_configs.py    CheXpert / VinDr-CXR / NIH-CXR paths and class labels
│   ├── optimizer_configs.py  Adam, SGD, schedulers, warmup
│   ├── eval_configs.py       OOD scoring configs (MCM, GL-MCM, energy, ...)
│   └── tuner_configs.py      Ray Tune settings
├── clip/             Vendored OpenAI CLIP (+ DualCoOp and GalLoP variants of the model)
├── models/           CLIP wrappers that add learnable prompts (BaseCLIP subclasses)
├── trainers/         Training/eval loops (BaseTrainer subclasses)
├── losses/           Method-specific loss functions
└── utils/
    ├── make.py        Factories: model, transforms, datasets, dataloaders, optimizer, scheduler
    ├── datasets.py    CXRDatasets, DictImageFolder
    ├── sampler.py     N-shot (few-shot) samplers
    ├── eval.py        AUC/ROC/PR, bootstrap CIs, OOD metrics
    ├── shuffle.py     GlobalBatchShuffler
    ├── logger.py      Logging setup
    └── utils.py       AverageMeter, config table printing
```

## How a run flows (`run.py`)

1. Hydra builds a `ProjectConf` from `+dataset=<name> +trainer=<name>`. The names are registered in `configs.py`.
2. `DATAPATH` is read from `.env` and used as the root for data, weights and outputs. Outputs go to `$DATAPATH/experiments_new/<dataset>/<trainer>[_<shots>]/<sampler>/exp_*/{logs,results,weights}`.
3. `utils/make.py` builds transforms, datasets and dataloaders.
4. `make_model` loads CLIP through `clip.load`, optionally loads CheXzero weights, and wraps it in the class registered in `BaseCLIP.registered_models[trainer.name]`.
5. The optimizer and scheduler are built. The trainer is looked up in `BaseTrainer.registered_trainers[trainer.name]`.
6. `trainer.train()` runs, then `trainer.evaluate(split='test')`.

`sweep.py` follows the same pipeline inside a Ray Tune trial. Config values written as lists (`+tune.optimizer="[adam, sgd]"`) become the search space.

## Registry pattern

A method is linked by one name string in four places:

| Piece | Where | Hook |
|---|---|---|
| Config | `allconfigs/train_configs.py` + `cs.store(...)` in `configs.py` | `+trainer=<name>` |
| Model | `models/<Method>.py` | `@BaseCLIP.register_model("<name>")` |
| Trainer | `trainers/<Method>.py` | `@BaseTrainer.register_trainer("<name>")` |
| Loss (optional) | `losses/<Method>.py` | returned from `get_losses()` |

The package `__init__.py` files import each module so the decorators run.

`BaseTrainer` provides the loop: AMP, checkpointing, evaluation and bootstrap. Subclasses override `get_losses`, `model_forward`, `compute_loss` and sometimes `forward_pass`. `LoCoOpTrainer` extends `CoOpTrainer`, and the `locoop` name reuses `CoOpCLIP`.

## Methods in the repo

| Trainer name | Model | Trainer | Loss |
|---|---|---|---|
| `zeroshot` | `models/zeroshot.py` | `trainers/zeroshot.py` | none |
| `coop` | `models/CoOp.py` | `trainers/CoOp.py` | cross-entropy |
| `locoop` | `models/CoOp.py` | `trainers/LoCoOp.py` | `losses/LoCoOp.py` |
| `dualcoop` | `models/DualCoOp.py` | `trainers/DualCoOp.py` | `losses/DualCoOp.py` |
| `gallop` | `models/GalLoP.py` | `trainers/GalLoP.py` | `losses/GalLoP.py` |
| `plot` | `models/PLOT.py` | `trainers/PLOT.py` | `losses/PLOT.py` |

## Adding a new method

1. Add a `FooConf(TrainConf)` dataclass in `allconfigs/train_configs.py`. Import it and call `cs.store(name='foo', group='trainer', node=FooConf)` in `configs.py`.
2. Add `models/Foo.py` with a `BaseCLIP` subclass registered as `"foo"`, then import it in `models/__init__.py`.
3. Add `trainers/Foo.py` with a `BaseTrainer` subclass registered as `"foo"`, then import it in `trainers/__init__.py`.
4. Run `python run.py +dataset=chexpert +trainer=foo`.

## Known inconsistencies

These are in the repo as it stands:

- `models/__init__.py` and `trainers/__init__.py` import modules that are not in the repo: `manual_prompt`, `MoCoOp`, `DualCoOpPP`, `Ensemble_prompt1/2/3` and `vision_classifier`. They need removing, or the files need adding, before `import models` or `import trainers` will succeed.
- `eval.py` imports `detectors`, which is also missing.
- `run.py` references `configs.weights_dir`, and the dataset configs have hardcoded absolute `zeroshot_results_dir` paths (`/mnt/...`). Adjust these to your machine.
- There is no `.gitignore` and no `.env` template. `.env` must define `DATAPATH`.
