
# Original Llama
 accelerate launch -m --main_process_port 12329 lm_eval \
        --model hf \
        --model_args pretrained=meta-llama/Llama-2-7b-hf,trust_remote_code=True \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
        --batch_size 8 \
        --log_samples \
        --output_path output/llama_7b_hf

# 50% sparsity
initial_steps=0
step_increment=1
max_iterations=1
is_first_training=1
targeted_sparsity=0.5

for ((i=1; i<=max_iterations; i++)); do
    current_steps=$((initial_steps + i * step_increment))  # Calculate the number of steps trained so far

    echo "Training for $current_steps steps..."
    deepspeed --master_port 12329 experiments/pretrain_sparse_model.py \
      --use_sparse_model --targeted_sparsity $targeted_sparsity \
      --set_sparsity_aware_threshold --print_sparsity \
      --use_wandb --max_steps $current_steps --model_save \
      --train_batch_size 1 --test_batch_size 4 --use_flash_attn --gradient_accumulation_steps 8 \
      --ds_config_path ds_config.json --max_seq_length 1024  \
      --output_dir /matx/u/lukeai/ckpt --is_first_training $is_first_training \
      --gradient_checkpointing \
      --model_name "sparse_llama_7b_hf2" \
      --base_model_repo_id "meta-llama/Llama-2-7b-hf" \
      --process_index 1

    is_first_training=0
    model_directory=$(cat model_directory1.txt)
    echo "model directory: $model_directory"

    echo "Evaluating after $current_steps steps..."
    accelerate launch -m --main_process_port 12329 lm_eval \
        --model hf \
        --model_args pretrained=$model_directory,trust_remote_code=True \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
        --batch_size 32 \
        --log_samples \
        --output_path output/llama_sparse_50p_${current_steps}steps2
done

# 70% sparsity
initial_steps=0
step_increment=1
max_iterations=1
is_first_training=1
targeted_sparsity=0.7

for ((i=1; i<=max_iterations; i++)); do
    current_steps=$((initial_steps + i * step_increment))  # Calculate the number of steps trained so far

    echo "Training for $current_steps steps..."
    deepspeed --master_port 12330 experiments/pretrain_sparse_model.py \
      --use_sparse_model --targeted_sparsity $targeted_sparsity \
      --set_sparsity_aware_threshold --model_save \
      --use_wandb --max_steps $current_steps  \
      --train_batch_size 1 --test_batch_size 4 --use_flash_attn --gradient_accumulation_steps 4 \
      --ds_config_path ds_config.json --max_seq_length 1024  \
      --output_dir /matx/u/lukeai/ckpt --is_first_training $is_first_training \
      --gradient_checkpointing \
      --model_name "sparse_llama_7b_hf2" \
      --base_model_repo_id "meta-llama/Llama-2-7b-hf" \
      --process_index 1

    is_first_training=0
    model_directory=$(cat model_directory1.txt)
    echo "model directory: $model_directory"

    echo "Evaluating after $current_steps steps..."
    accelerate launch -m --main_process_port 12329 lm_eval \
        --model hf \
        --model_args pretrained=$model_directory,trust_remote_code=True \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
        --batch_size 32 \
        --log_samples \
        --output_path output/llama_sparse_70p_${current_steps}steps2
done

# 90% sparsity
initial_steps=0
step_increment=1
max_iterations=1
is_first_training=1
targeted_sparsity=0.9

for ((i=1; i<=max_iterations; i++)); do
    current_steps=$((initial_steps + i * step_increment))  # Calculate the number of steps trained so far

    echo "Training for $current_steps steps..."
    deepspeed --master_port 12330 experiments/pretrain_sparse_model.py \
      --use_sparse_model --targeted_sparsity $targeted_sparsity \
      --set_sparsity_aware_threshold --model_save \
      --use_wandb --max_steps $current_steps \
      --train_batch_size 1 --test_batch_size 4 --use_flash_attn --gradient_accumulation_steps 4 \
      --ds_config_path ds_config.json --max_seq_length 1024  \
      --output_dir /matx/u/lukeai/ckpt --is_first_training $is_first_training \
      --gradient_checkpointing \
      --model_name "sparse_llama_7b_hf2" \
      --base_model_repo_id "meta-llama/Llama-2-7b-hf" \
      --process_index 1

    is_first_training=0
    model_directory=$(cat model_directory1.txt)
    echo "model directory: $model_directory"

    echo "Evaluating after $current_steps steps..."
    accelerate launch -m --main_process_port 12329 lm_eval \
        --model hf \
        --model_args pretrained=$model_directory,trust_remote_code=True \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
        --batch_size 32 \
        --log_samples \
        --output_path output/llama_sparse_90p_${current_steps}steps2
done


# ReLUfication
initial_steps=0
step_increment=1
max_iterations=1
is_first_training=1

for ((i=1; i<=max_iterations; i++)); do
    current_steps=$((initial_steps + i * step_increment))  # Calculate the number of steps trained so far

    echo "Training for $current_steps steps..."
    deepspeed --master_port 12329 experiments/pretrain_sparse_model.py \
      --set_sparsity_aware_threshold --model_save \
      --use_sparse_model \
      --use_wandb --max_steps $current_steps \
      --train_batch_size 1 --test_batch_size 2 --use_flash_attn \
      --gradient_accumulation_steps 4 --ds_config_path ds_config.json \
      --max_seq_length 1024  \
      --output_dir /matx/u/lukeai/ckpt \
      --gradient_checkpointing \
      --is_first_training $is_first_training \
      --model_name "relu_llama_7b_hf2" \
      --base_model_repo_id "meta-llama/Llama-2-7b-hf" \
      --use_relu --process_index 1

    is_first_training=0
    model_directory=$(cat model_directory1.txt)
    echo "model directory: $model_directory"

    echo "Evaluating after $current_steps steps..."
    accelerate launch -m --main_process_port 12329 lm_eval --model hf \
        --model_args pretrained=$model_directory,trust_remote_code=True \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
        --batch_size 16 \
        --log_samples \
        --output_path output/relu_llama_${current_steps}steps
done





