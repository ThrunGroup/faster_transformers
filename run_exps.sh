#!/bin/bash

deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.9 \
--is_plot --set_sparsity_aware_threshold --print_sparsity
deepspeed experiments/mistral_instruct_tuning.py --use_wandb
