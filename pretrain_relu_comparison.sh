#!/bin/bash

source /sailhome/vxbrando/.bashrc
conda activate mistral

# Initial steps and increment
initial_steps=0
step_increment=500
max_iterations=4  # This will train up to 2000 steps in increments of 500


for ((i=1; i<=max_iterations; i++)); do
    current_steps=$((initial_steps + i * step_increment))  # Calculate the current max_steps

    echo "Training for $current_steps steps..."
    deepspeed --master_port 12311 experiments/pretrain_sparse_model.py --use_sparse_model --use_relu \
    --set_sparsity_aware_threshold --print_sparsity --use_wandb --num_epochs 1 --max_steps $current_steps --model_save \
    --train_batch_size 4 --test_batch_size 4 --use_flash_attn --gradient_accumulation_steps 2 \
    --ds_config_path ds_config.json --push_to_hub --resume_from_checkpoint --max_seq_length 512
p
    echo "Evaluating after $current_steps steps..."
    accelerate launch -m --main_process_port 31212 lm_eval --model hf \
        --model_args pretrained=/scr/jay/ckpt/Mistral_Sparse_refined_web_50p,trust_remote_code=True \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
        --batch_size auto \
        --log_samples \
        --output_path output/mistral_sparse_refined_relu_${current_steps}steps
done

for ((i=1; i<=max_iterations; i++)); do
    current_steps=$((initial_steps + i * step_increment))  # Calculate the current max_steps

    echo "Training for $current_steps steps..."
    deepspeed --master_port 12311 experiments/pretrain_sparse_model.py --use_sparse_model --targeted_sparsity 0.7 \
    --set_sparsity_aware_threshold --print_sparsity --use_wandb --num_epochs 1 --max_steps $current_steps --model_save \
    --train_batch_size 1 --test_batch_size 1 --use_flash_attn --gradient_accumulation_steps 8 \
    --ds_config_path ds_config.json --push_to_hub --resume_from_checkpoint --max_seq_length 100

    echo "Evaluating after $current_steps steps..."
    accelerate launch -m --main_process_port 31212 lm_eval --model hf \
        --model_args pretrained=/scr/jay/ckpt/Mistral_Sparse_refined_web_50p,trust_remote_code=True \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
        --batch_size auto \
        --log_samples \
        --output_path output/mistral_sparse_refined_50p_${current_steps}steps
done

for ((i=1; i<=max_iterations; i++)); do
    current_steps=$((initial_steps + i * step_increment))  # Calculate the current max_steps

    echo "Training for $current_steps steps..."
    deepspeed --master_port 12311 experiments/pretrain_sparse_model.py --use_sparse_model --targeted_sparsity 0.9 \
    --set_sparsity_aware_threshold --print_sparsity --use_wandb --num_epochs 1 --max_steps $current_steps --model_save \
    --train_batch_size 1 --test_batch_size 1 --use_flash_attn --gradient_accumulation_steps 8 \
    --ds_config_path ds_config.json --push_to_hub --resume_from_checkpoint --max_seq_length 100

    echo "Evaluating after $current_steps steps..."
    accelerate launch -m --main_process_port 31212 lm_eval --model hf \
        --model_args pretrained=/scr/jay/ckpt/Mistral_Sparse_refined_web_50p,trust_remote_code=True \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
        --batch_size auto \
        --log_samples \
        --output_path output/mistral_sparse_refined_50p_${current_steps}steps
done


