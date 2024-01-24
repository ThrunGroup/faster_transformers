#!/bin/bash

deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.5 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.7 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.85 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.9 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.95 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.99 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.5 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --use_sparse_regularization
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.7 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --use_sparse_regularization
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --use_sparse_regularization
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.85 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --use_sparse_regularization
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.9 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --use_sparse_regularization
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.95 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --use_sparse_regularization
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.99 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --use_sparse_regularization


deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.5 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --dataset_type cola
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.7 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --dataset_type cola
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --dataset_type cola
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.85 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --dataset_type cola
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.9 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --dataset_type cola
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.95 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --dataset_type cola
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.99 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --dataset_type cola
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.5 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --use_sparse_regularization \
--dataset_type cola
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.7 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --use_sparse_regularization \
--dataset_type cola
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --use_sparse_regularization \
--dataset_type cola
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.85 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --use_sparse_regularization --dataset_type cola
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.9 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --use_sparse_regularization --dataset_type cola
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.95 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --use_sparse_regularization --dataset_type cola
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.99 \
--set_sparsity_aware_threshold --print_sparsity --train_batch_size 4 --test_batch_size 4 --use_sparse_regularization --dataset_type cola
