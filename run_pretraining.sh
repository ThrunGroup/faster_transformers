#!/bin/bash

# # Sparse Mistral Pretrained with 30K Steps
# deepspeed experiments/mistral_instruct_tuning.py --use_wandb --do_pretraining --use_sparse_model --targeted_sparsity 0.8 \
# --is_plot --set_sparsity_aware_threshold --print_sparsity --max_steps 5000 \
# --train_batch_size 8 --test_batch_size 16 \
# --gradient_checkpointing True --push_to_hub True --model_name "sparse_sparse_80_percent_pretraining_warmup_20K_steps_5k" \
# --use_gradual_sparsification --num_warmup_steps 2000 --increment_ratio 0.15


# Sparse Mistral Pretrained with 30K Steps
deepspeed experiments/mistral_instruct_tuning.py --use_wandb --do_pretraining \
--is_plot --max_steps 5000 \
--train_batch_size 8 --test_batch_size 16 \
--gradient_checkpointing True --push_to_hub True --model_name "sparse_sparse_80_percent_pretraining_warmup_20K_steps_5k" \
--use_gradual_sparsification --num_warmup_steps 2000 --increment_ratio 0.2