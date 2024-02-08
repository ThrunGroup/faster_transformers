#!/bin/bash
# dataset=("sst2")
# seeds=(1 2 3)

# for data in "${dataset[@]}"; do
#     for seed in "${seeds[@]}"; do
#         # Sparse Mistral Pretrained with 10K Steps
#         deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
#         --is_plot --set_sparsity_aware_threshold --print_sparsity --num_epochs 15 --max_steps 1  \
#         --train_batch_size 4 --test_batch_size 8 --dataset_type $data \
#         --gradient_checkpointing True --push_to_hub False --model_name "mistral_sparse_80_percent_$data_1000" \
#         --pretrained_model_to_load "Mistral_Sparse_pretraining_80_percent_10000" --seed $seed

#         # Original Mistral
#         deepspeed experiments/mistral_instruct_tuning.py --use_wandb  --num_epochs 15  \
#         --train_batch_size 4 --test_batch_size 8 --dataset_type $data \
#         --gradient_checkpointing True --push_to_hub False --model_name "original_glue_$data" --seed $seed
#     done
# done

dataset=("cola" "wic" "boolq" "qnli")
seeds=(1)

for data in "${dataset[@]}"; do
    for seed in "${seeds[@]}"; do
       # Original Mistral
        deepspeed experiments/mistral_instruct_tuning.py --use_wandb  --do_evaluate \
        --train_batch_size 4 --test_batch_size 64 --dataset_type $data \
        --gradient_checkpointing True --model_name "original_glue_$data" --seed $seed

        # # Sparse Mistral Pretrained with 10K Steps
        # deepspeed experiments/mistral_instruct_tuning.py --use_wandb --use_sparse_model --targeted_sparsity 0.8 \
        # --is_plot --set_sparsity_aware_threshold --print_sparsity --do_evaluate \
        # --train_batch_size 4 --test_batch_size 32 --dataset_type $data \
        # --gradient_checkpointing True --model_name "mistral_sparse_80_percent_$data_1000" \
        # --pretrained_model_to_load "Mistral_Sparse_pretraining_80_percent_10000" --seed $seed
    done
done
