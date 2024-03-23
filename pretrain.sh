#!/bin/bash

source /sailhome/vxbrando/.bashrc
conda activate mistral
# Initial steps and increment
initial_steps=1
step_increment=250
max_iterations=5 # This will train up to 2000 steps in increments of 500
targeted_sparsity=0.5
date=$(date +%Y-%m-%d)
sparsity_percentage=$(printf "%.0f" $(echo "$targeted_sparsity * 100" | bc))

echo "Sparsity Percentage: $sparsity_percentage"

deepspeed --master_port 12310 experiments/pretrain_sparse_model.py --use_sparse_model --targeted_sparsity $targeted_sparsity \
    --set_sparsity_aware_threshold --print_sparsity --use_wandb --num_epochs 11 --max_steps $initial_steps --model_save \
    --train_batch_size 1 --test_batch_size 1 --use_flash_attn --gradient_accumulation_steps 4 \
    --ds_config_path ds_config.json --max_seq_length 1024 --is_plot --gradient_checkpointing \
    --output_dir /scr/jay/ckpt/$date --push_to_hub

echo "Evaluating after $initial_steps steps..."
accelerate launch -m --main_process_port 12310 lm_eval --model hf \
--model_args pretrained=/scr/jay/ckpt/$date/Mistral_Sparse_refined_web_${sparsity_percentage}p_no_adapter,trust_remote_code=True \
--tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
--batch_size auto \
--log_samples \
--output_path /scr/jay/output/$date/mistral_sparse_refined_${sparsity_percentage}p_${initial_steps}steps

for ((i=1; i<=max_iterations; i++)); do
    current_steps=$((initial_steps + i * step_increment))  # Calculate the current max_steps

    echo "Training for $current_steps steps..."
    deepspeed --master_port 12310 experiments/pretrain_sparse_model.py --use_sparse_model --targeted_sparsity $targeted_sparsity \
    --set_sparsity_aware_threshold --print_sparsity --use_wandb --num_epochs 11 --max_steps $current_steps --model_save \
    --train_batch_size 1 --test_batch_size 1 --use_flash_attn --gradient_accumulation_steps 4 \
    --ds_config_path ds_config.json --resume_from_checkpoint --max_seq_length 1024 --gradient_checkpointing \
    --output_dir /scr/jay/ckpt/$date --push_to_hub

    echo "Evaluating after $current_steps steps..."
    accelerate launch -m --main_process_port 12310 lm_eval --model hf \
        --model_args pretrained=/scr/jay/ckpt/$date/Mistral_Sparse_refined_web_${sparsity_percentage}p_no_adapter,trust_remote_code=True \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
        --batch_size auto \
        --log_samples \
        --output_path /scr/jay/output/$date/mistral_sparse_refined_${sparsity_percentage}p_${current_steps}steps
done
