#!/bin/bash

# Initial steps and increment
initial_steps=20
step_increment=20
max_iterations=2  # This will train up to 2000 steps in increments of 500
date=$(date +%Y-%m-%d)

deepspeed --master_port 12310 experiments/pretrain_sparse_model.py --use_sparse_model --targeted_sparsity 0.7 \
    --set_sparsity_aware_threshold --print_sparsity --use_wandb --num_epochs 11 --max_steps $initial_steps --model_save \
    --train_batch_size 1 --test_batch_size 1 --use_flash_attn --gradient_accumulation_steps 4 \
    --ds_config_path ds_config.json --max_seq_length 1024 --is_plot --gradient_checkpointing \
    --output_dir /matx/u/lukeai/ckpt --push_to_hub --model_name "sparse_mistral"

for ((i=1; i<=max_iterations; i++)); do
    current_steps=$((initial_steps + i * step_increment))  # Calculate the current max_steps

    echo "Training for $current_steps steps..."
    deepspeed --master_port 12310 experiments/pretrain_sparse_model.py --use_sparse_model --targeted_sparsity 0.7 \
    --set_sparsity_aware_threshold --print_sparsity --use_wandb --num_epochs 11 --max_steps $current_steps --model_save \
    --train_batch_size 1 --test_batch_size 1 --use_flash_attn --gradient_accumulation_steps 4 \
    --ds_config_path ds_config.json --push_to_hub --resume_from_checkpoint --max_seq_length 1024 --gradient_checkpointing \
    --output_dir /matx/u/lukeai/ckpt --model_name "sparse_mistral"
done


