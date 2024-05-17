#!/bin/bash

# 50% sparsity
initial_steps=1
step_increment=0
max_iterations=1
is_first_training=1
targeted_sparsity=0.5

for ((i=1; i<=max_iterations; i++)); do
    current_steps=$((initial_steps + i * step_increment))

    echo "Training for $current_steps steps..."
    deepspeed --master_port 12329 experiments/pretrain_sparse_model.py \
      --use_sparse_model --targeted_sparsity $targeted_sparsity \
      --set_sparsity_aware_threshold --print_sparsity \
      --max_steps $current_steps \
      --train_batch_size 1 --test_batch_size 4 --use_flash_attn --gradient_accumulation_steps 8 \
      --ds_config_path ds_config.json --max_seq_length 1024  \
      --output_dir $1 --is_first_training $is_first_training \
      --gradient_checkpointing \
      --model_name "sparse_llama_7b_hf2" \
      --base_model_repo_id "meta-llama/Llama-2-7b-hf"
done

# 70% sparsity
initial_steps=1
step_increment=0
max_iterations=1
is_first_training=1
targeted_sparsity=0.7

for ((i=1; i<=max_iterations; i++)); do
    current_steps=$((initial_steps + i * step_increment))

    echo "Training for $current_steps steps..."
    deepspeed --master_port 12329 experiments/pretrain_sparse_model.py \
      --use_sparse_model --targeted_sparsity $targeted_sparsity \
      --set_sparsity_aware_threshold --print_sparsity \
      --max_steps $current_steps \
      --train_batch_size 1 --test_batch_size 4 --use_flash_attn --gradient_accumulation_steps 8 \
      --ds_config_path ds_config.json --max_seq_length 1024  \
      --output_dir $1 --is_first_training $is_first_training \
      --gradient_checkpointing \
      --model_name "sparse_llama_7b_hf2" \
      --base_model_repo_id "meta-llama/Llama-2-7b-hf"
done

# 90% sparsity
initial_steps=1
step_increment=0
max_iterations=1
is_first_training=1
targeted_sparsity=0.9

for ((i=1; i<=max_iterations; i++)); do
    current_steps=$((initial_steps + i * step_increment))

    echo "Training for $current_steps steps..."
    deepspeed --master_port 12329 experiments/pretrain_sparse_model.py \
      --use_sparse_model --targeted_sparsity $targeted_sparsity \
      --set_sparsity_aware_threshold --print_sparsity \
      --max_steps $current_steps \
      --train_batch_size 1 --test_batch_size 4 --use_flash_attn --gradient_accumulation_steps 8 \
      --ds_config_path ds_config.json --max_seq_length 1024  \
      --output_dir $1 --is_first_training $is_first_training \
      --gradient_checkpointing \
      --model_name "sparse_llama_7b_hf2" \
      --base_model_repo_id "meta-llama/Llama-2-7b-hf"
done

