#!/bin/bash

# Initial steps and increment
initial_steps=0
step_increment=200
max_iterations=5  # This will train up to 2000 steps in increments of 500
is_first_training=1

for ((i=1; i<=max_iterations; i++)); do
    current_steps=$((initial_steps + i * step_increment))  # Calculate the number of steps trained so far

#    echo "Training for $current_steps steps..."
#    deepspeed --master_port 12329 experiments/pretrain_sparse_model.py --use_sparse_model --targeted_sparsity 0.7 \
#    --set_sparsity_aware_threshold --print_sparsity --use_wandb --max_steps $current_steps --model_save \
#    --train_batch_size 1 --test_batch_size 1 --use_flash_attn --gradient_accumulation_steps 8 \
#    --ds_config_path ds_config.json --push_to_hub --max_seq_length 1024  \
#    --output_dir /matx/u/lukeai/ckpt --is_first_training $is_first_training --gradient_checkpointing\
#    --model_name "sparse_llama_7b_hf_test" --base_model_repo_id "meta-llama/Llama-2-7b-hf" \

#    is_first_training=0
    model_directory=$(cat model_directory.txt)
    model_directory=/matx/u/lukeai/ckpt/sparse_llama_7b_refined_web_50p_no_adapter
#    echo "model directory: $model_directory"

    echo "Evaluating after $current_steps steps..."
    accelerate launch -m --main_process_port 12329 lm_eval --model hf \
        --model_args pretrained=$model_directory,trust_remote_code=True \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
        --batch_size 16 \
        --log_samples \
        --output_path output/llama_sparse_50p_${current_steps}steps_test
done


