#!/bin/bash

# original 
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --max_steps 1

# # Sparse SiLU
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
# --is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 16

# # Pretraining on openwebtext
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --do_pretraining --use_sparse_model --targeted_sparsity 0.8 \
# --is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 1 --max_steps 10000 --train_batch_size 8 \
# --gradient_checkpointing True --push_to_hub True --model_name "Mistral_Sparse_pretraining_80_percent_10000" \

# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --do_pretraining --use_sparse_model --targeted_sparsity 0.8 \
# --is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 1 --max_steps 3000 --train_batch_size 8 \
# --gradient_checkpointing True --push_to_hub True --model_name "Mistral_Sparse_pretraining_80_percent_3000"

# Zeroshot with the pretrained model
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
--is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 10  \
--train_batch_size 2 --test_batch_size 4 --dataset_type "boolq" \
--gradient_checkpointing True --push_to_hub False --model_name "mistral_sparse_80_percent_boolq_1000" \
--pretrained_model_to_load "Mistral_Sparse_pretraining_80_percent_10000"

deepspeed experiments/mistral_instruct_tuning.py --use_wandb  --num_epochs 10  \
--train_batch_size 2 --test_batch_size 4 --dataset_type "boolq" \
--gradient_checkpointing True --push_to_hub False --model_name "original_glue_boolq"

# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
# --is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 10 --eval_steps 25  \
# --train_batch_size 8 --test_batch_size 16 \
# --gradient_checkpointing True --push_to_hub True --model_name "mistral_sparse_80_percent_cola_raw" \

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