#!/bin/bash

source /sailhome/vxbrando/.bashrc
conda activate mistral

deepspeed --master_port 12311 experiments/pretrain_sparse_model.py --use_sparse_model --targeted_sparsity 0.70 \
--set_sparsity_aware_threshold --print_sparsity --use_wandb --num_epochs 1 --max_steps 1000 --model_save \
--train_batch_size 1 --test_batch_size 1 --use_flash_attn --gradient_accumulation_steps 8 \
--ds_config_path ds_config.json --push_to_hub --gradient_checkpointing


