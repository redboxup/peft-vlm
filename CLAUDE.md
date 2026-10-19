# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Prompt-learning PEFT methods (CoOp, LoCoOp, DualCoOp, GalLoP, PLOT) for CLIP-based medical VLMs, benchmarked on chest X-ray datasets (CheXpert, VinDr-CXR, NIH-CXR). Configuration uses Hydra and sweeps use Ray Tune. See `README.md` for usage and results, and `docs/architecture.md` for a directory map.

## Commands

There is no test suite, linter, or build step. The environment is Conda (`environment.yml`, env name `peftvlm`), and CUDA is required (`BaseTrainer` raises without it).

```bash
python3 preprocess.py --dataset-name chexpert|vincxr|nihcxr --raw-folder-path <raw> --out-path <out> --resolution 320 --num-workers 16
python3 run.py +dataset=chexpert +trainer=coop            # single experiment
python3 run.py -m +dataset=chexpert,vincxr +trainer=zeroshot   # Hydra multirun
python3 sweep.py -m +trainer=<t> +dataset=<d> tune.num_samples=N tune.gpus_per_trial=G +tune.optimizer="[adam, sgd]" +tune.trainer.batch_size="[16,32,64]"
```

`run.py` reads `DATAPATH` from a `.env` file in the working directory. It is the root for data, CLIP/CheXzero weights (`weights_dir`) and outputs (`experiments_new/<dataset>/<trainer>[_shots]/<sampler>/exp_*/{logs,results,weights}`). The `.env` is not tracked.

## Architecture

**Name-keyed registries tie each method together.** The trainer name (e.g. `coop`) is used in four places, and all of them must agree:
1. A config dataclass in `allconfigs/train_configs.py`, registered with `cs.store(name=..., group='trainer', ...)` in `configs.py`.
2. A model registered with `@BaseCLIP.register_model(name)` in `models/`. It wraps the vendored CLIP from `clip/` and adds learnable prompts.
3. A trainer registered with `@BaseTrainer.register_trainer(name)` in `trainers/`.
4. An optional loss in `losses/`, returned by the trainer's `get_losses()`.

Registration happens at import time. `models/__init__.py` and `trainers/__init__.py` must import each module for its decorators to run.

**Pipeline** (`run.py`, mirrored in `sweep.py` and `eval.py`): `ProjectConf` from Hydra → `utils/make.py` factories (`make_transforms`, `make_datasets`, `make_dataloaders`, `make_model`, `make_optimizer`, `make_scheduler`) → `BaseTrainer.registered_trainers[name](...)` → `train()` → `evaluate('test')`. `make_model` always starts from pretrained CLIP via `clip.load(..., model_i=trainer.name)`, so `clip/model*.py` hold per-method model variants. It then optionally loads CheXzero weights.

**Trainers** subclass `BaseTrainer` (AMP, checkpointing, per-label AUC evaluation in `trainers/base.py`) and override `get_losses`, `model_forward`, `compute_loss`. `LoCoOpTrainer` extends `CoOpTrainer`, and `locoop` shares `CoOpCLIP`.

**Sweeps:** in `sweep.py`, config values given as lists (`+tune.<path>="[a, b]"`) are flattened into a Ray Tune search space. Everything else is fixed.

## Known state of the repo

- `models/__init__.py` and `trainers/__init__.py` import modules that don't exist (`manual_prompt`, `MoCoOp`, `DualCoOpPP`, `Ensemble_prompt1/2/3`, `vision_classifier`), and `eval.py` imports a missing `detectors` package. Importing these packages fails until the imports are pruned or the files restored. Several configs in `configs.py` have no matching model or trainer.
- `allconfigs/dataset_configs.py` hardcodes machine-specific `/mnt/...` paths in `zeroshot_results_dir`.
- `run.py`'s `verify_dataloaders` block references unrelated datasets (`irma`, `mura`, `boneage`) and the nonexistent `configs.trainer.sampler`. Don't enable it as is.

## Hooks

`.claude/settings.json` runs three project checks (`.claude/hooks/`): `py_compile` on edited Python, a registry consistency check on edits to `configs.py`/`models/`/`trainers/`, and a machine-specific-path check on edits under `allconfigs/`. A `git add`/`git commit` is blocked if it would stage weights or files over 10 MB.

The registry check compares against `.claude/hooks/registry_baseline.json`, which records the drift listed above so only *new* drift blocks. After fixing (or deliberately adding) a registration, run `python3 .claude/hooks/check_registry.py --update-baseline` and commit the result.
