#!/bin/bash
dataset=("wic" "qnli")
seeds=(1 2 3)

for data in "${dataset[@]}"; do
    for seed in "${seeds[@]}"; do
        # Sparse Mistral Pretrained with 10K Steps
        deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
        --is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 15  \
        --train_batch_size 2 --test_batch_size 4 --dataset_type $data \
        --gradient_checkpointing True --push_to_hub False --model_name "mistral_sparse_80_percent_$data_1000" \
        --pretrained_model_to_load "Mistral_Sparse_pretraining_80_percent_10000" --seed $seed

        # Original Mistral
        deepspeed experiments/mistral_instruct_tuning.py --use_wandb  --num_epochs 15  \
        --train_batch_size 2 --test_batch_size 4 --dataset_type $data \
        --gradient_checkpointing True --push_to_hub False --model_name "original_glue_$data" --seed $seed
    done
done
