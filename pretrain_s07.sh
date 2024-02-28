#!/bin/bash

source /sailhome/vxbrando/.bashrc
conda activate mistral
echo "Evaluating after $initial_steps steps..."
# Initial steps and increment
initial_steps=100
step_increment=500
max_iterations=10  # This will train up to 2000 steps in increments of 500
date=$(date +%Y-%m-%d)

deepspeed --master_port 12310 experiments/pretrain_sparse_model.py --use_sparse_model --targeted_sparsity 0.7 \
    --set_sparsity_aware_threshold --print_sparsity --use_wandb --num_epochs 11 --max_steps $initial_steps --model_save \
    --train_batch_size 1 --test_batch_size 1 --use_flash_attn --gradient_accumulation_steps 4 \
    --ds_config_path ds_config.json --max_seq_length 1024 --is_plot --gradient_checkpointing \
    --output_dir /scr/jay/ckpt/$date

echo "Evaluating after $initial_steps steps..."
accelerate launch -m --main_process_port 12310 lm_eval --model hf \
--model_args pretrained=/scr/jay/ckpt/$date/Mistral_Sparse_refined_web_70p,trust_remote_code=True \
--tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
--batch_size auto \
--log_samples \
--num_fewshot 5 \
--output_path /scr/jay/output/$date/mistral_sparse_refined_70p_${initial_steps}steps

for ((i=1; i<=max_iterations; i++)); do
    current_steps=$((initial_steps + i * step_increment))  # Calculate the current max_steps

    echo "Training for $current_steps steps..."
    deepspeed --master_port 12310 experiments/pretrain_sparse_model.py --use_sparse_model --targeted_sparsity 0.7 \
    --set_sparsity_aware_threshold --print_sparsity --use_wandb --num_epochs 11 --max_steps $current_steps --model_save \
    --train_batch_size 1 --test_batch_size 1 --use_flash_attn --gradient_accumulation_steps 4 \
    --ds_config_path ds_config.json --push_to_hub --resume_from_checkpoint --max_seq_length 1024 --gradient_checkpointing \
    --output_dir /scr/jay/ckpt/$date

    echo "Evaluating after $current_steps steps..."
    accelerate launch -m --main_process_port 12310 lm_eval --model hf \
        --model_args pretrained=/scr/jay/ckpt/$date/Mistral_Sparse_refined_web_70p,trust_remote_code=True \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
        --batch_size auto \
        --log_samples \
        --num_fewshot 5 \
        --output_path /scr/jay/output/$date/mistral_sparse_refined_70p_${current_steps}steps
done


