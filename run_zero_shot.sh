#!/bin/bash

dataset=("cola" "wic" "boolq" "qnli")
seeds=(1)

for data in "${dataset[@]}"; do
    for seed in "${seeds[@]}"; do
        # Original Mistral
        deepspeed experiments/mistral_instruct_tuning.py --do_evaluate \
            --train_batch_size 4 --test_batch_size 64 --dataset_type "$data" \
            --gradient_checkpointing True --model_name "original_glue_$data" \
            --use_wandb --is_debugging

        # Sparse Mistral Pretrained with 10K Steps
        deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
            --is_plot --set_sparsity_aware_threshold --print_sparsity --do_evaluate \
            --train_batch_size 4 --test_batch_size 32 --dataset_type "$data" \
            --gradient_checkpointing True --model_name "mistral_sparse_80_percent_10000_zero_shot_$data" \
            --pretrained_model_to_load "Mistral_Sparse_pretraining_80_percent_10000" --seed "$seed"
    done
done
