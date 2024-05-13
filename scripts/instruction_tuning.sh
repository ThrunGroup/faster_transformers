#!/bin/bash

## Define an array of targeted sparsity values
targeted_sparsity_values=(0 0.5 0.7 0.9)
dataset_types=(cola sst2)

for dataset in "${dataset_types[@]}"; do
  for sparsity in "${targeted_sparsity_values[@]}"; do
      # Check if sparsity is greater than 0 to decide on using sparse model
      if (( $(echo "$sparsity > 0" | bc -l) )); then
          # Command for sparse models
          deepspeed experiments/instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity $sparsity \
          --set_sparsity_aware_threshold --print_sparsity --train_batch_size 8 --test_batch_size 8 --dataset_type $dataset \
          --gradient_accumulation_steps 2 --num_epochs 8 --gradient_accumulation_steps 2
      else
          # Command for non-sparse model
          deepspeed experiments/instruct_tuning.py --use_wandb --train_batch_size 8 --test_batch_size 8 \
          --dataset_type $dataset --gradient_accumulation_steps 2
      fi
  done
  deepspeed experiments/instruct_tuning.py --use_wandb --use_sparse_model --use_relu \
  --set_sparsity_aware_threshold --print_sparsity --train_batch_size 8 --test_batch_size 8 --dataset_type $dataset \
  --gradient_accumulation_steps 2 --num_epochs 8
done

dataset_types=(boolq)
for dataset in "${dataset_types[@]}"; do
  for sparsity in "${targeted_sparsity_values[@]}"; do
      # Check if sparsity is greater than 0 to decide on using sparse model
      if (( $(echo "$sparsity > 0" | bc -l) )); then
          # Command for sparse models
          deepspeed experiments/instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity $sparsity \
          --set_sparsity_aware_threshold --print_sparsity --train_batch_size 2 --test_batch_size 2 --dataset_type $dataset \
          --gradient_accumulation_steps 2 --num_epochs 8 --gradient_accumulation_steps 8
      else
          # Command for non-sparse model
          deepspeed experiments/instruct_tuning.py --use_wandb --train_batch_size 2 --test_batch_size 2 \
          --dataset_type $dataset --gradient_accumulation_steps 8
      fi
  done
  deepspeed experiments/instruct_tuning.py --use_wandb --use_sparse_model --use_relu \
  --set_sparsity_aware_threshold --print_sparsity --train_batch_size 2 --test_batch_size 2 --dataset_type $dataset \
  --gradient_accumulation_steps 8 --num_epochs 8
done

