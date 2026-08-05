# python3 sweep.py -m verbose=False \
#     +trainer=dualcoop++ \
#     trainer.few_shots=2,4,8,16 \
#     trainer.few_shot_sampler=random,stratified \
#     +dataset=chexpert,vincxr,nihcxr \
#     tune.num_samples=40 \
#     tune.cpus_per_trial=1 \
#     tune.gpus_per_trial=0.25 \
#     +tune.optimizer="[adam, sgd]" \
#     +tune.trainer.epochs="[25, 50]" \
#     +tune.trainer.batch_size="[16, 32, 64]" \
#     +tune.warmup.use_warmup="[True, False]" \
#     +tune.scheduler="[cosine, linear, none]" \
#     +tune.trainer.gamma_pos="[0.5, 1, 1.5, 2.0, 2.5, 3.0, 4.0, 5, 6]" \
#     +tune.trainer.gamma_neg="[0.5, 1, 1.5, 2.0, 2.5, 3.0, 4.0, 5, 6]" \
#     +tune.trainer.gamma="[0.5, 1, 1.5, 2.0, 2.5, 3.0, 4.0, 5, 6, 7]" \
#     +tune.trainer.learning_rate="[1e-1, 1e-4]"

# CUDA_VISIBLE_DEVICES=0 python3 sweep.py -m verbose=False \
#     trainer.move_to_host=True \
#     +trainer=dualcoop \
#     trainer.few_shots=16 \
#     trainer.few_shot_sampler=random,stratified \
#     +dataset=chexpert,vincxr,nihcxr \
#     tune.num_samples=40 \
#     tune.cpus_per_trial=1 \
#     tune.gpus_per_trial=0.15 \
#     +tune.optimizer="[adam, sgd]" \
#     +tune.trainer.epochs="[25, 50]" \
#     +tune.trainer.batch_size="[16, 32, 64]" \
#     +tune.warmup.use_warmup="[True, False]" \
#     +tune.scheduler="[cosine, linear, none]" \
#     +tune.trainer.gamma_pos="[0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]" \
#     +tune.trainer.gamma_neg="[0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]" \
#     +tune.trainer.learning_rate="[1e-1, 1e-4]" \

# python3 sweep.py -m verbose=False \
#     trainer.move_to_host=True \
#     +trainer=locoop \
#     trainer.few_shots=2,4,8,16 \
#     +dataset=chexpert,vincxr,nihcxr \
#     tune.num_samples=40 \
#     tune.cpus_per_trial=1 \
#     tune.gpus_per_trial=0.10 \
#     +tune.optimizer="[adam, sgd]" \
#     +tune.trainer.epochs="[25, 50]" \
#     +tune.trainer.batch_size="[16, 32, 64]" \
#     +tune.warmup.use_warmup="[True, False]" \
#     +tune.scheduler="[cosine, linear, none]" \
#     +tune.trainer.en="[0.25, 0.5, 0.75, 1.0, 1.5, 2, 2.5, 3]" \
#     +tune.trainer.top_k="[10, 15, 20, 25, 30]" \
#     +tune.trainer.learning_rate="[1e-1, 1e-4]" \

# python3 eval.py -m +dataset=chexpert,vincxr,nihcxr \
#     +trainer=dualcoop++ \
#     trainer.few_shots=2,4,8,16 \
#     +eval=mcm \
#     eval.num_workers=16 \
#     eval.top_k=3 \

# python3 eval.py -m +dataset=chexpert,vincxr,nihcxr \
#     +trainer=dualcoop \
#     trainer.few_shots=2,4,8,16 \
#     +eval=mcm \
#     eval.num_workers=16 \
# #     eval.top_k=3 \

# python3 eval.py -m +dataset=chexpert,vincxr,nihcxr \
#     +trainer=locoop \
#     trainer.few_shots=2,4,8,16 \
#     +eval=mcm,glmcm \
#     eval.num_workers=16 \
#     eval.top_k=3 \

#---------------------------CoOp---------------------------
# python3 sweep.py -m verbose=False \
#     trainer.move_to_host=True \
#     +trainer=coop \
#     trainer.few_shots=2,4,8,16 \
#     +dataset=chexpert \
#     tune.num_samples=25 \
#     tune.cpus_per_trial=1 \
#     tune.gpus_per_trial=0.10 \
#     +tune.optimizer="[adam, sgd]" \
#     +tune.trainer.epochs="[25, 50]" \
#     +tune.trainer.batch_size="[16, 32, 64]" \
#     +tune.warmup.use_warmup="[True, False]" \
#     +tune.scheduler="[cosine, linear, none]" \
#     +tune.trainer.learning_rate="[1e-1, 1e-4]" \

# python3 eval.py -m +dataset=chexpert \
#     +trainer=coop \
#     trainer.few_shots=2,4,8,16 \
#     +eval=mcm \
#     eval.num_workers=16 \
#     eval.top_k=3 \


# python3 sweep.py -m verbose=False \
#     trainer.move_to_host=True \
#     +trainer=gallop \
#     trainer.few_shots=16 \
#     trainer.few_shot_sampler=multiclass \
#     +dataset=chexpert,vincxr,nihcxr \
#     tune.num_samples=30 \
#     tune.cpus_per_trial=1 \
#     tune.gpus_per_trial=0.23 \
#     +tune.optimizer="[adam, sgd]" \
#     +tune.trainer.epochs="[25, 50]" \
#     +tune.trainer.batch_size="[16, 32, 64]" \
#     +tune.warmup.use_warmup="[True, False]" \
#     +tune.scheduler="[cosine, linear, none]" \
#     +tune.trainer.base_size="[5,6]" \
#     +tune.trainer.n_global_prompts="[1, 2, 3, 4, 5, 6, 7]" \
#     +tune.trainer.n_local_prompts="[1, 2, 3, 4, 5, 6, 7]" \
#     +tune.trainer.learn_local_prompt="[True]" \
#     +tune.trainer.learn_global_prompt="[True]" \
#     +tune.trainer.use_global_loss="[True, False]" \
#     +tune.trainer.use_local_loss="[True, False]" \
#     +tune.trainer.learning_rate="[1e-1, 1e-4]" \

# python3 eval.py -m +dataset=chexpert,vincxr,nihcxr \
#     +trainer=gallop \
#     trainer.few_shots=2,4,8,16 \
#     +eval=mcm \
#     eval.num_workers=16 \
#     eval.top_k=3 \

#---------------------------Ensemble Prompt 1 (without KL Divergence)---------------------------
# python3 sweep.py -m verbose=False \
#     trainer.move_to_host=True \
#     +trainer=ensemble_prompt1 \
#     trainer.few_shots=2,4,8,16 \
#     trainer.sampler=random,stratified \
#     +dataset=chexpert \
#     tune.num_samples=30 \
#     tune.cpus_per_trial=1 \
#     tune.gpus_per_trial=0.15 \
#     +tune.optimizer="[adam, sgd]" \
#     +tune.trainer.epochs="[25, 50]" \
#     +tune.trainer.batch_size="[4, 8, 16, 32, 64]" \
#     +tune.trainer.n_prompts="[1]" \
#     +tune.warmup.use_warmup="[True, False]" \
#     +tune.scheduler="[cosine, linear, none]" \
#     +tune.trainer.alpha="[0.1, 0.2, 0.4, 0.6, 0.8, 1]" \
#     +tune.trainer.gamma="[0.5, 1, 1.5, 2, 2.5, 3.5, 4.0, 4.5, 5]" \
#     +tune.trainer.learning_rate="[1e-1, 1e-4]" \

# python3 eval.py -m +dataset=chexpert \
#     +trainer=ensemble_prompt1 \
#     trainer.few_shots=2,4,8,16 \
#     +trainer.few_shot_sampler=random,stratified \
#     +eval=mcm \
#     eval.num_workers=16 \
#     eval.top_k=3 \

#---------------------------Ensemble Prompt 2 (with KL Divergence)---------------------------

# python3 sweep.py -m verbose=False \
#     trainer.move_to_host=True \
#     +trainer=ensemble_prompt2 \
#     trainer.few_shots=2,4,8,16 \
#     trainer.sampler=random,stratified \
#     +dataset=chexpert \
#     tune.num_samples=30 \
#     tune.cpus_per_trial=1 \
#     tune.gpus_per_trial=0.15 \
#     +tune.optimizer="[adam, sgd]" \
#     +tune.trainer.epochs="[25, 50, 100, 150, 200]" \
#     +tune.trainer.batch_size="[4, 8, 16, 32, 64]" \
#     +tune.trainer.n_prompts="[2]" \
#     +tune.warmup.use_warmup="[True, False]" \
#     +tune.scheduler="[cosine, linear, none]" \
#     +tune.trainer.alpha="[0.1, 0.2, 0.4, 0.6, 0.8, 1]" \
#     +tune.trainer.gamma="[0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 2, 4.0, 5]" \
#     +tune.trainer.kl_div="[0.1, 0.4, 0.6, 0.8, 1, 1.5, 2]" \
#     +tune.trainer.learning_rate="[1e-1, 1e-5]" \

# python3 eval.py -m +dataset=chexpert \
#     +trainer=ensemble_prompt2 \
#     trainer.few_shots=2,4,8,16 \
#     +trainer.few_shot_sampler=random,stratified \
#     +eval=mcm \
#     eval.num_workers=16 \
#     eval.top_k=3 \

# ---------------------------Prompt Embedding Visualization---------------------------
# python3 prompt_embedding_visualization.py -m +dataset=chexpert \
#     +trainer=ensemble_prompt2 \
#     trainer.few_shots=16 \
#     trainer.sampler=random,stratified \
#     +eval=mcm \
#     eval.num_workers=16 \
#     eval.top_k=1 \

#---------------------------Manual Prompt---------------------------

# python3 sweep.py -m verbose=False \
#     trainer.move_to_host=True \
#     +trainer=manual_prompt \
#     trainer.few_shots=0 \
#     +dataset=chexpert \
#     tune.num_samples=3 \
#     tune.cpus_per_trial=1 \
#     tune.gpus_per_trial=0.15 \
#     +tune.optimizer="[adam, sgd]" \
#     +tune.trainer.epochs="[25, 50]" \
#     +tune.trainer.batch_size="[4, 8, 16, 32, 64]" \

# python3 eval.py -m +dataset=chexpert \
#     +trainer=manual_prompt \
#     trainer.few_shots=0 \
#     +eval=mcm \
#     eval.num_workers=16 \
#     eval.top_k=3 \


#---------------------------Ensemble Prompt 3 (with KL Divergence)---------------------------

# python3 sweep.py -m verbose=False \
#     trainer.move_to_host=True \
#     +trainer=ensemble_prompt3 \
#     trainer.few_shots=16 \
#     trainer.sampler=stratified \
#     +dataset=chexpert \
#     tune.num_samples=25 \
#     tune.cpus_per_trial=1 \
#     tune.gpus_per_trial=0.15 \
#     +tune.optimizer="[adam, sgd]" \
#     +tune.trainer.epochs="[25, 50, 100, 150, 200]" \
#     +tune.trainer.batch_size="[4, 8, 16, 32, 64]" \
#     +tune.trainer.n_prompts="[2]" \
#     +tune.warmup.use_warmup="[True, False]" \
#     +tune.scheduler="[cosine, linear, none]" \
#     +tune.trainer.alpha="[0.1, 0.2, 0.4, 0.6, 0.8, 1]" \
#     +tune.trainer.gamma="[0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 2, 4.0, 5]" \
#     +tune.trainer.kl_div="[0.1, 0.4, 0.6, 0.8, 1, 1.5, 2]" \
#     +tune.trainer.learning_rate="[1e-1, 1e-5]" \

python3 eval.py -m +dataset=chexpert \
    +trainer=ensemble_prompt3 \
    trainer.few_shots=2,4,8,16 \
    trainer.few_shot_sampler=random,stratified \
    +eval=mcm \
    eval.num_workers=16 \
    eval.top_k=3 \



