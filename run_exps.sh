#!/bin/bash

deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.9 \
--is_plot --set_sparsity_aware_threshold --print_sparsity
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --output_dir /scr/zgh23
deepspeed experiments/mistral_instruct_tuning.py --dataset_type boolq --use_wandb --print_sparsity \
--set_sparsity_aware_threshold  --is_plot --use_sparse_model --train_batch_size 4 --test_batch_size 4
