import math
import dataclasses
from typing import List
from pathlib import Path
from dataclasses import dataclass, field
from math import inf


@dataclass
class TrainConf:
    name: str = dataclasses.field(init=False)
    img_size: int = 224
    transforms: str = 'weak'
    batch_size: int = 4
    epochs: int = 10
    learning_rate: float = 1e-4
    context_length: int = 77
    few_shot_sampler: str = "multiclass"  # "multiclass", "random"
    few_shots: int = dataclasses.field(init=False)

    logging: bool = True  # maintain logs
    move_to_host: bool = False
    cache_dataset: bool = False
    num_workers: int = 0
    save_interval: int = 100
    log_interval: int = 10
    metric: str = dataclasses.field(init=False)
    results_dir: Path = dataclasses.field(init=False)

    pretrained: bool = True
    model_weights: str = dataclasses.field(init=False)
    prec: str = 'fp32'  # fp16, fp32, amp

    # method specific
    # N_CTX: int = dataclasses.field(init=False)
    # CTX_INIT: str = dataclasses.field(init=False)
    # CLASS_TOKEN_POSITION: str = dataclasses.field(init=False)
    # CSC: bool = dataclasses.field(init=False)

@dataclass
class ZeroShotConf(TrainConf):
    name: str = 'zeroshot'
    n_prompts: int = 1  # number of prompts
    batch_size: int = 64
    pos_context: str = 'chest X-ray demonstrates {}'
    neg_context: str = 'no {}'
    model_weights: str = 'best_128_0.0002_original_15000_0.859.pt'
    metric: str = 'mAUC'
    few_shots: int = 0
    learn_local_prompt: bool = False  # learn local prompts


@dataclass
class CoOpConf(TrainConf):
    name: str = 'coop'
    N_CTX: int = 16  # number of context vectors
    CTX_INIT: str = "chest X-ray demonstrates"  # initialisation words
    CTX_INIT_NEG: str = "no findings of"
    CLASS_TOKEN_POSITION: str = 'end'  # 'front', 'end', 'middle'
    CSC: bool = False  # class-specific context
    model_weights: str = 'best_128_0.0002_original_15000_0.859.pt'
    few_shot_sampler: str = "multiclass"  # "multiclass"
    few_shots: int = 16
    metric: str = 'mAUC'


@dataclass
class LoCoOpConf(TrainConf):
    name: str = 'locoop'
    ce: float = 1.0  # weight to ce loss
    en: float = 0.25  # weight to ood regularisation
    top_k: int = 10  # top-k values to be used in ood regularisation
    N_CTX: int = 16  # number of context vectors
    CTX_INIT: str = ""  # initialisation words
    CLASS_TOKEN_POSITION: str = 'end'  # 'front', 'end', 'middle'
    CSC: bool = False  # class-specific context
    model_weights: str = 'best_128_0.0002_original_15000_0.859.pt'
    few_shot_sampler: str = "multiclass"
    few_shots: int = 16
    metric: str = 'mAUC'


@dataclass
class DualCoOpConf(TrainConf):
    name: str = 'dualcoop'
    N_CTX: int = 16
    CTX_INIT: str = ""
    CTX_INIT_POS: str = ""  # initialisation words for positive context
    CTX_INIT_NEG: str = "no"  # initialisation words for negative context
    CLASS_TOKEN_POSITION: str = 'end'
    CSC: bool = False  # class-specific context
    model_weights: str = 'best_128_0.0002_original_15000_0.859.pt'
    few_shot_sampler: str = "stratified"  # "stratified", "random"
    few_shots: int = 16
    metric: str = 'mAUC'
    gamma_pos: float = 1
    gamma_neg: float = 4


@dataclass
class DualCoOpPPConf(TrainConf):
    name: str = 'dualcoop++'
    N_CTX: int = 16
    CTX_INIT: str = ""
    CTX_INIT_EVI: str = ""  # initialisation words for positive context
    CTX_INIT_POS: str = ""  # initialisation words for positive context
    CTX_INIT_NEG: str = ""  # initialisation words for negative context
    CLASS_TOKEN_POSITION: str = 'end'
    CSC: bool = False  # class-specific context
    model_weights: str = 'best_128_0.0002_original_15000_0.859.pt'
    few_shot_sampler: str = "stratified"  # "stratified", "random"
    few_shots: int = 16
    metric: str = 'mAUC'
    gamma: float = 1.5
    gamma_pos: float = 1
    gamma_neg: float = 4


@dataclass
class MoCoOpConf(TrainConf):
    name: str = 'mocoop'
    N_CTX: int = 16  # number of context vectors
    CTX_INIT: str = ""
    CTX_INIT_POS: str = ""  # initialisation words for positive context
    CTX_INIT_NEG: str = ""  # initialisation words for negative context
    CLASS_TOKEN_POSITION: str = 'end'  # 'front', 'end', 'middle'
    CSC: bool = False  # class-specific context
    model_weights: str = 'best_128_0.0002_original_15000_0.859.pt'
    few_shot_sampler: str = "random"
    few_shots: int = 16
    metric: str = 'mAUC'


@dataclass
class PLOTConf(TrainConf):
    name: str = 'plot'
    N_CTX: int = 16
    N_Prompt: int = 4
    CTX_INIT: str = ""  # initialisation words
    CLASS_TOKEN_POSITION: str = 'end'  # 'front', 'end', 'middle'
    CSC: bool = False  # class-specific context
    model_weights: str = 'best_128_0.0002_original_15000_0.859.pt'
    few_shot_sampler: str = "multiclass"
    few_shots: int = 16
    metric: str = 'mAUC'

@dataclass
class GalLoPConf(TrainConf):
    name: str = 'gallop'
    N_CTX: int = 16 # number of context vectors
    CTX_INIT: str = "this is a photo of {}" # initialisation words
    CLASS_TOKEN_POSITION: str = 'end'  # 'front', 'end', 'middle'
    CSC: bool = False  # class
    learn_local_proj: bool = True  # learn local projection
    learn_local_prompt: bool = True  # learn local prompts
    learn_global_prompt: bool = True  # learn global prompt
    n_global_prompts: int = 4  # number of global prompts
    n_local_prompts: int = 4  # number of local prompts
    use_global_loss: bool = True
    use_local_loss: bool = True
    base_size: int = 5 
    prompts_batch_size: int = 10000  # how many prompts to use in a batch
    identity_init: bool = True  # initialize linear layer with identity matrix
    model_weights: str = 'best_128_0.0002_original_15000_0.859.pt'
    few_shots: int = 16
    few_shot_sampler: str = "multiclass" 
    metric: str = 'mAUC'

@dataclass
class ManualPromptConf(TrainConf):
    name: str = 'manual_prompt'
    n_prompts: int = 1  # number of prompts
    batch_size: int = 64
    pos_context: str = "chest X-ray demonstrates"
    neg_context: str = "no findings of"
    model_weights: str = 'best_128_0.0002_original_15000_0.859.pt'
    metric: str = 'mAUC'
    few_shots: int = 0
    learn_local_prompt: bool = False  # learn local prompts
    few_shot_sampler: str = "multiclass"  # "multiclass"

@dataclass
class Ensemble_PromptConf1(TrainConf):
    name: str = 'ensemble_prompt1'
    N_CTX: int = 16
    n_prompts: int = 2 # number of prompts
    few_shots: int = 16
    CTX_INIT: str = ""
    Vector_initialization: str = "manual_prompt" # "random", "manual_prompt", "manual_prompt_const_vector"
    CLASS_TOKEN_POSITION: str = 'end'
    CSC: bool = False
    model_weights: str = 'best_128_0.0002_original_15000_0.859.pt'
    metric: str = 'mAUC'
    few_shot_sampler: str = "stratified"  # "stratified", "random"
    few_shots: int = 16
    alpha: float = 0.25
    gamma: float = 2.0
    gamma_pos: float = 1.0
    gamma_neg: float = 4.0

@dataclass
class Vision_ClassifierConf(TrainConf):
    name: str = 'vision_classifier'
    model_weights: str = 'best_128_0.0002_original_15000_0.859.pt'
    dropout:float = 0.2
    few_shots: int = 1000  # use all data
    metric: str = 'mAUC'
    few_shot_sampler: str = "random"
    vision_enc_dim: int = 768
    alpha: float = 0.25
    gamma: float = 2.0
    gamma_pos: float = 1.0
    gamma_neg: float = 4.0

@dataclass
class Ensemble_PromptConf2(TrainConf):
    name: str = 'ensemble_prompt2'
    N_CTX: int = 16
    n_prompts: int = 1 # number of prompts
    few_shots: int = 16
    CTX_INIT: str = ""
    Vector_initialization: str = "manual_prompt" # "random", "manual_prompt", "manual_prompt_const_vector"
    CLASS_TOKEN_POSITION: str = 'end'
    CSC: bool = False
    model_weights: str = 'best_128_0.0002_original_15000_0.859.pt'
    metric: str = 'mAUC'
    few_shot_sampler: str = "stratified"  # "stratified", "random"
    alpha: float = 0.4
    gamma: float = 0.5
    kl_div: float = 0.3
    focal: float = 1

@dataclass
class Ensemble_PromptConf3(TrainConf):
    name: str = 'ensemble_prompt3'
    N_CTX: int = 16
    n_prompts: int = 2 # number of prompts
    prompt_split_ratio: list = (0.4, 0.6)
    few_shots: int = 16
    CTX_INIT: str = ""
    Vector_initialization: str = "manual_prompt" # "random", "manual_prompt", "manual_prompt_const_vector"
    CLASS_TOKEN_POSITION: str = 'end'
    CSC: bool = False
    model_weights: str = 'best_128_0.0002_original_15000_0.859.pt'
    metric: str = 'mAUC'
    few_shot_sampler: str = "stratified"  # "stratified", "random"
    alpha: float = 0.4
    gamma: float = 0.5
    kl_div: float = 0.3
    focal: float = 1