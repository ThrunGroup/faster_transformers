#!/bin/bash

# original 
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --max_steps 1

# # Sparse SiLU
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
# --is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 16

# # Pretraining on openwebtext
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --do_pretraining --use_sparse_model --targeted_sparsity 0.8 \
# --is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 1 --max_steps 1000 --train_batch_size 8 \
# --gradient_checkpointing True --push_to_hub True --model_name "Mistral_Sparse_pretraining_80_percent"

# Zeroshot with the pretrained model
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
--is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 1 --max_steps 1 --train_batch_size 8 \
--gradient_checkpointing True --push_to_hub False --model_name "Mistral_Sparse_pretraining_80_percent" \
--pretrained_model_to_load "thrunlab/Mistral_Sparse_pretraining_80_percent"

# # Sparse SiLU with Warmup
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.9 \
# --is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 1 --max_steps 500

# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
# --is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 16 --max_steps 2 \
# --gradient_checkpointing True --push_to_hub True --model_name "loading_test"

# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
# --is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 16 --max_steps 2 \
# --gradient_checkpointing True --push_to_hub False --model_name "loading_test" \
# --pretrained_model_to_load "thrunlab/loading_test"

# Sparse SiLU
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
# --is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 8

# Sparse SiLU with Warmup
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
# --is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 8 \
# --use_gradual_sparsification --num_warmup_steps 450 --increment_ratio 1.0

# Sparse SiLU with Warmup
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
# --is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 8 \
# --use_gradual_sparsification --num_warmup_steps 450 --increment_ratio 0.5

# ReLU
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --print_sparsity --use_relu --num_epochs 16