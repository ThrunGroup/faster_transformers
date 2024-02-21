#!/bin/bash

# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --train_batch_size 2 --test_batch_size 2
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.99 \
# --set_sparsity_aware_threshold --print_sparsity --train_batch_size 2 --test_batch_size 2 --use_sparse_regularization

source /sailhome/vxbrando/.bashrc
conda activate mistral

#deepspeed experiments/pretrain_sparse_model.py --use_sparse_model --targeted_sparsity 0.70 \
#--set_sparsity_aware_threshold --print_sparsity --use_wandb --num_epochs 1 --gradient_checkpointing --model_save \
#--train_batch_size 1 --test_batch_size 1 --use_flash_attn --max_steps 20000 --gradient_accumulation_steps 8

## Define an array of targeted sparsity values
targeted_sparsity_values=(0 0.5 0.7 0.8 0.9)
dataset_types=('cola', 'sst2')

for dataset in "${dataset_types[@]}"; do
  for sparsity in "${targeted_sparsity_values[@]}"; do
      # Check if sparsity is greater than 0 to decide on using sparse model
      if (( $(echo "$sparsity > 0" | bc -l) )); then
          # Command for sparse models
          deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity $sparsity \
          --set_sparsity_aware_threshold --print_sparsity --train_batch_size 8 --test_batch_size 8 --dataset_type $dataset\
          --gradient_accumulation_steps 2 --num_epochs 8 --use_flash_attn
      else
          # Command for non-sparse model (if you have any)
          deepspeed experiments/mistral_instruct_tuning.py --use_wandb --train_batch_size 8 --test_batch_size 8 \
          --dataset_type $dataset --use_flash_attn
      fi
  done
done

for sparsity in "${targeted_sparsity_values[@]}"; do
    # Check if sparsity is greater than 0 to decide on using sparse model
    if (( $(echo "$sparsity > 0" | bc -l) )); then
        # Command for sparse models
        deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity $sparsity \
        --set_sparsity_aware_threshold --print_sparsity --train_batch_size 2 --test_batch_size 2 --dataset_type boolq\
        --gradient_accumulation_steps 2 --num_epochs 8 --use_flash_attn
    else
        # Command for non-sparse model (if you have any)
        deepspeed experiments/mistral_instruct_tuning.py --use_wandb --train_batch_size 2 --test_batch_size 2 \
        --dataset_type boolq --use_flash_attn
    fi
done

#deepspeed experiments/mistral_instruct_tuning.py --use_wandb --train_batch_size 1 --test_batch_size 4 \
#--dataset_type boolq
#
#
#deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model\
#          --set_sparsity_aware_threshold --print_sparsity --train_batch_size 8 --test_batch_size 8 --dataset_type sst2\
#          --gradient_accumulation_steps 2 --num_epochs 7 --use_relu
#
#deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model\
#          --set_sparsity_aware_threshold --print_sparsity --train_batch_size 8 --test_batch_size 8 --dataset_type cola\
#          --gradient_accumulation_steps 2 --num_epochs 15 --use_relu


#
#
#
#targeted_sparsity_values=(0.8 0.9)
#for sparsity in "${targeted_sparsity_values[@]}"; do
#    # Check if sparsity is greater than 0 to decide on using sparse model
#    if (( $(echo "$sparsity > 0" | bc -l) )); then
#        # Command for sparse models
#        deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity $sparsity \
#        --set_sparsity_aware_threshold --print_sparsity --train_batch_size 1 --test_batch_size 4 --dataset_type boolq \
#        --use_flash_attn
#    else
#        # Command for non-sparse model (if you have any)
#        deepspeed experiments/mistral_instruct_tuning.py --use_wandb --train_batch_size 1 --test_batch_size 4 \
#        --dataset_type boolq --use_flash_attn
#    fi
#done
#
#deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity $sparsity \
#        --set_sparsity_aware_threshold --print_sparsity --train_batch_size 1 --test_batch_size 4 --dataset_type boolq \
#        --use_flash_attn --use_relu
#
#deepspeed experiments/mistral_instruct_tuning.py --use_wandb --train_batch_size 1 --test_batch_size 4 \
#--dataset_type boolq --use_sparse_model --set_sparsity_aware_threshold --use_relu

