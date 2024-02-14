#!/bin/bash

dataset=(cola wic sst2)
seeds=(1)

for data in "${dataset[@]}"; do
  for seed in "${seeds[@]}"; do

    # Sparse Mistral with Graceful Reg
    deepspeed --master_port 30001 experiments/mistral_instruct_tuning.py \
    --use_wandb \
    --use_sparse_model \
    --targeted_sparsity 0.8 \
    --is_plot \
    --set_sparsity_aware_threshold \
    --print_sparsity \
    --num_epochs 10 \
    --train_batch_size 8 \
    --test_batch_size 64 \
    --dataset_type "$data" \
    --model_name "mistral_sparse_80_percent_\"$data\"_graceful_reg_300steps " \
    --seed "$seed" \
    --use_sparse_regularization \
    --use_graceful_regularization \
    --num_warmup_steps 300 \

#    # Sparse Mistral with Graceful Reg & Keep Reg
#    deepspeed --master_port 30001 experiments/mistral_instruct_tuning.py \
#    --use_wandb \
#    --use_sparse_model \
#    --targeted_sparsity 0.8 \
#    --is_plot \
#    --set_sparsity_aware_threshold \
#    --print_sparsity \
#    --num_epochs 10 \
#    --train_batch_size 8 \
#    --test_batch_size 32 \
#    --dataset_type "$data" \
#    --model_name "mistral_sparse_80_percent_\"$data\"_graceful_reg_keep_reg" \
#    --seed "$seed" \
#    --use_sparse_regularization \
#    --use_graceful_regularization \
#    --keep_regularization_with_kill \
#    --num_warmup_steps 100 \
#
#    # Sparse Mistral with Graceful Reg & Keep Reg & Distillation
#    deepspeed --master_port 30001 experiments/mistral_instruct_tuning.py \
#    --use_wandb \
#    --use_sparse_model \
#    --targeted_sparsity 0.8 \
#    --is_plot \
#    --set_sparsity_aware_threshold \
#    --print_sparsity \
#    --num_epochs 10 \
#    --train_batch_size 8 \
#    --test_batch_size 32 \
#    --dataset_type "$data" \
#    --model_name "mistral_sparse_80_percent_\"$data\"_graceful_reg_distill" \
#    --seed "$seed" \
#    --use_sparse_regularization \
#    --use_graceful_regularization \
#    --keep_regularization_with_kill \
#    --num_warmup_steps 100 \
#    --use_distillation \

#    deepspeed  --master_port 30002  experiments/mistral_instruct_tuning.py \
#    --use_wandb \
#    --use_sparse_model \
#    --targeted_sparsity 0.8 \
#    --is_plot \
#    --set_sparsity_aware_threshold \
#    --print_sparsity \
#    --num_epochs 10 \
#    --train_batch_size 8 \
#    --test_batch_size 32 \
#    --dataset_type "$data" \
#    --gradient_checkpointing True \
#    --model_name "mistral_sparse_80_percent_\"$data\"_no_gradual_sparsification" \
#    --seed "$seed" \

    deepspeed  --master_port 30002  experiments/mistral_instruct_tuning.py \
    --use_wandb \
    --num_epochs 10 \
    --train_batch_size 8 \
    --test_batch_size 32 \
    --dataset_type "$data" \
    --gradient_checkpointing True \
    --model_name "mistral_original_\"$data\"" \
    --seed "$seed"
  done
done

