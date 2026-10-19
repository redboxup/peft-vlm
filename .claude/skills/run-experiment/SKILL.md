---
name: run-experiment
description: Launch a single experiment (run.py) or a Ray Tune sweep (sweep.py) and locate its results. Use when the user asks to train, evaluate, benchmark or sweep a method on a dataset.
---

Preflight (fail fast, report what is missing instead of guessing):
- `nvidia-smi` works - `BaseTrainer` raises without CUDA.
- `.env` exists in the repo root and defines `DATAPATH` (see `.env.example`). Do not print its contents.
- Preprocessed data exists under `$DATAPATH/data/preprocessed/<dataset>/` (paths come from `allconfigs/dataset_configs.py`); if not, run `preprocess.py` (see README).
- CLIP/CheXzero weights exist at `$DATAPATH/<weights_dir>/<trainer.model_weights>` (default `weights/chexzero_weights/CheXzero_Models`).
- `python3 -c "import models, trainers"` succeeds (known broken imports: see CLAUDE.md).

Run (from repo root):
```bash
python3 run.py +dataset=<chexpert|vincxr|nihcxr> +trainer=<zeroshot|coop|locoop|dualcoop|gallop|plot>
python3 run.py -m +dataset=chexpert,vincxr +trainer=coop          # multirun
python3 sweep.py -m +trainer=<t> +dataset=<d> tune.num_samples=N tune.gpus_per_trial=G \
    +tune.optimizer="[adam, sgd]" +tune.trainer.batch_size="[16,32,64]"
```
Training is long; run in the background and check the log rather than blocking.

Outputs: `$DATAPATH/experiments_new/<dataset>/<trainer>[_<shots>]/<sampler>/exp_00000000/{logs,results,weights}`. `run.py` hardcodes `exp_00000000`, so a re-run overwrites the previous one - copy results first if they must be kept. The key result is `results/ID_test_auc.csv` (columns `label`, `auc`); the headline number is the mean AUC (mAUC) printed at the end of the run.
