#!/bin/bash

# original 
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --is_debugging --num_epochs 1

# # # Sparse SiLU
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.9 \
--is_plot --set_sparsity_aware_threshold --print_sparsity  --is_debugging --num_epochs 1

# # Sparse SiLU with Warmup
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.9 \
# --is_plot --set_sparsity_aware_threshold --print_sparsity --is_debugging --num_epochs 1 \
# --gradual_sparsification --num_warmup_steps 2 --increment_ratio 0.5