# Parameter-Efficient Adaptation for Medical Foundation Models

A Pytorch project that implements prompt-learning based Parameter Efficient Fine-Tuning (PEFT) method for medical foundation vision-language models.
State-of-art PEFT methods are implemented and evaluated on multiple chest X-ray datasets, including CheXpert, VinDr-CXR, and NIH-CXR.

Beyond the proposed method, the repository is designed as a modular, object-oriented research framework with reusable components for datasets, trainers, models, losses, and evaluation.
It supports scalable experimentation through Hydra configuration management and Ray Tune distributed hyperparameter optimization, enabling reproducible experiments, parallel multi-GPU sweeps, and efficient benchmarking of existing and newly developed prompt learning methods.

## Getting Started

### Installation

Clone the repository:

```bash
git clone https://github.com/redboxup/peft-vlm.git
cd peft-vlm
```

Create the Conda environment from the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate peftvlm
```

### Preprocessing Datasets

Preprocess a dataset using:

```bash
python3 preprocess.py \
    --dataset-name <dataset_name> \
    --raw-folder-path <raw_dataset_path> \
    --out-path <output_path> \
    --resolution 320 \
    --num-workers 16
```

#### Parameters

- `--dataset-name <dataset_name>`: Dataset to preprocess (e.g., `chexpert`, `vincxr`, `nihcxr`).
- `--raw-folder-path <raw_dataset_path>`: Directory containing the raw dataset.
- `--out-path <output_path>`: Directory where the preprocessed dataset will be saved.
- `--resolution 320`: Image resolution used during preprocessing (default: `320`).
- `--num-workers 16`: Number of worker processes used for preprocessing (default: `16`).

### Running Experiments

Run an experiment using:

```bash
python3 run.py \
    +dataset=<dataset_name> \
    +trainer=<trainer_name>
```

#### Parameters

- `+dataset=<dataset_name>`: Dataset to evaluate (e.g., `chexpert`, `vincxr`, `nihcxr`).
- `+trainer=<trainer_name>`: Training method to execute (e.g., `zeroshot`, `coop`).


#### Running Multiple Configurations

Hydra's multirun mode (`-m`) can be used to evaluate multiple configurations simultaneously.

```bash
python3 run.py -m \
    +dataset=chexpert,vincxr \
    +trainer=zeroshot \
```

The above command evaluates both datasets using the `zeroshot` trainer.

## Supported Methods

This repository provides unified implementations of several state-of-the-art prompt learning methods for adapting CLIP-based medical vision-language models. All methods share a common preprocessing, training, and evaluation pipeline, enabling fair and reproducible benchmarking across multiple chest X-ray datasets.

| Method | Venue | Trainer |
|---------|:-----:|:-------:|
| CoOp | [CVPR 2022](https://arxiv.org/abs/2203.05557) | `coop` |
| DualCoOp | [NeurIPS 2022](https://arxiv.org/pdf/2206.09541) | `dualcoop` |
| LoCoOp | [NeurIPS 2023](https://arxiv.org/abs/2306.01293) | `locoop` |
| GalLoP | [ECCV 2024](https://arxiv.org/abs/2407.01400) | `gallop` |
| PLoT | [ICLR 2023](https://arxiv.org/abs/2604.01837) | `plot` |


### Quantitative Comparison

Results are reported as **Mean (SD)** over 5 runs. 

| Method | CheXpert mAUC ↑ | Head ↑ | Tail ↑ | VinDr-CXR mAUC ↑ | Head ↑ | Tail ↑ |
|:-------|----------------:|-------:|-------:|-----------------:|--------:|--------:|
| Zero-shot | 73.50 | 83.41 | 66.10 | 77.80 | 76.02 | 78.97 |
| CoOp | 72.3 (1.9) | 74.5 (3.3) | 70.7 (1.0) | 64.9 (1.4) | 66.5 (3.4) | 64.0 (0.2) |
| LoCoOp | 70.9 (7.3) | 77.4 (5.4) | 66.1 (8.8) | 71.7 (1.6) | 68.8 (5.6) | 73.5 (0.6) |
| GalLoP | 62.5 (1.8) | 76.0 (1.6) | 52.4 (2.3) | 72.3 (0.1) | 69.3 (0.6) | 74.1 (1.2) |
| PLOT | 70.5 (2.3) | 75.4 (0.9) | 66.9 (4.8) | 54.9 (0.5) | 67.6 (1.1) | 68.9 (2.1) |
| DualCoOp | 71.8 (1.3) | 80.3 (0.9) | 65.5 (1.7) | 83.3 (0.1) | 82.0 (0.2) | 84.0 (1.3) |


## Distributed Hyperparameter Optimization

The repository uses **Ray Tune** together with **Hydra** to perform scalable hyperparameter optimization.

Hyperparameter search spaces are defined by assigning candidate values enclosed in square brackets (`[]`). During optimization, Ray Tune samples configurations from these search spaces and executes trials in parallel.

For example,

```text
+tune.optimizer="[adam, sgd]"
```

searches over both optimizers, while

```text
+tune.trainer.batch_size="[16, 32, 64]"
```

searches over three candidate batch sizes.

A typical optimization run is

```bash
python sweep.py -m \
    +trainer= <trainer_name> \
    +dataset= <dataset_name> \
    tune.num_samples= <num_samples> \
    tune.gpus_per_trial= <gpu_mem_per_trial> \
    +tune.optimizer="[adam, sgd]" \
    +tune.trainer.batch_size="[16,32,64]"
```

### Multi-GPU Execution

Ray Tune automatically schedules trials across all available CPU and GPU resources.

The number of resources allocated to each trial can be controlled using

```text
tune.cpus_per_trial
tune.gpus_per_trial
```

allowing multiple experiments to execute concurrently on a single machine or across multiple GPUs.

## Citation

If you find this repository useful in your research, please consider citing it:

```bibtex
@misc{anudeep2026multiscale,
  author       = {A. S. Anudeep},
  title        = {Parameter-Efficient Adaptation for Medical Foundation Models},
  year         = {2026},
  howpublished = {\url{https://github.com/redboxup/peft-vlm}},
  note         = {GitHub repository}
}
```