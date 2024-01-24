#!/bin/bash

# original 
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --num_epochs 8

# Sparse SiLU
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.9 \
--is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 8

# Sparse SiLU with Warmup
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.9 \
--is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 8 \
--gradual_sparsification --num_warmup_steps 20 --increment_ratio 0.5

# Sparse SiLU
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
--is_plot --set_sparsity_aware_threshold --print_sparsity   --num_epochs 8

# Sparse SiLU with Warmup
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
--is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 8 \
--gradual_sparsification --num_warmup_steps 20 --increment_ratio 0.5