#!/bin/bash

# # original 
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --num_epochs 16

# # Sparse SiLU
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
# --is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 16

# Sparse SiLU with Warmup
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
--is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 16 \
--use_gradual_sparsification --num_warmup_steps 200 --increment_ratio 0.4

deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
--is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 16 \
--use_gradual_sparsification --num_warmup_steps 200 --increment_ratio 1.0

# Sparse SiLU
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
# --is_plot --set_sparsity_aware_threshold --print_sparsity   --num_epochs 8

# Sparse SiLU with Warmup
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
# --is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 8 \
# --use_gradual_sparsification --num_warmup_steps 450 --increment_ratio 1.0

# Sparse SiLU with Warmup
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
# --is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 8 \
# --use_gradual_sparsification --num_warmup_steps 450 --increment_ratio 0.5

# ReLU
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --print_sparsity --use_relu --num_epochs 16