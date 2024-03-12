#!/bin/bash


deepspeed experiments/pretrain_sparse_model.py --use_sparse_model --targeted_sparsity 0.70 \
--set_sparsity_aware_threshold --print_sparsity --use_wandb --num_epochs 1 --gradient_checkpointing --model_save \
--train_batch_size 1 --test_batch_size 1 --use_flash_attn --max_steps 20000 --gradient_accumulation_steps 8
