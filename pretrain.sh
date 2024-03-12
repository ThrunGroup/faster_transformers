#!/bin/bash

dataset_types="sst2,cola"
initial_steps=0
step_increment=200
max_iterations=10  # This will train up to 2000 steps in increments of 500

for ((i=0; i<max_iterations; i++)); do
    current_steps=$((initial_steps + i * step_increment))  # Calculate the current max_steps
    end_steps=$((current_steps + step_increment))
    model_name="Mistral_Sparse_${end_steps}"

    echo "Training for $current_steps steps..."
    echo "End steps: $end_steps"
    deepspeed --master_port 12312 experiments/pretrain_sparse_model.py --use_sparse_model --targeted_sparsity 0.5 \
    --train_batch_size 1 --test_batch_size 1 --use_flash_attn --gradient_accumulation_steps 4 \
    --ds_config_path ds_config.json --push_to_hub --use_sparse_regularization \
    --start_steps $current_steps --max_steps $step_increment \
    --use_graceful_regularization --end_steps $end_steps --model_save

    echo "Evaluating after $current_steps steps..."
    deepspeed --master_port 12312 experiments/evaluate_model.py --use_sparse_model --targeted_sparsity 0.5 \
    --train_batch_size 1 --test_batch_size 1 --use_flash_attn --gradient_accumulation_steps 4 \
    --ds_config_path ds_config.json --push_to_hub --use_sparse_regularization \
    --do_evaluate  --dataset_types $dataset_types --pretrained_model_to_load $model_name
done


