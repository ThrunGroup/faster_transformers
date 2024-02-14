#!/bin/bash
# Sparse Mistral Pretrained with 30K Steps
# Sparse Mistral with Graceful Reg
deepspeed --master_port 30002 experiments/mistral_instruct_tuning.py \
--do_pretraining True \
--use_wandb \
--use_sparse_model \
--targeted_sparsity 0.8 \
--is_plot \
--set_sparsity_aware_threshold \
--print_sparsity \
--train_batch_size 4 \
--test_batch_size 16 \
--max_steps 10000 \
--model_name "mistral_sparse_80_percent_\"$data\"_graceful_reg_300steps " \
--use_sparse_regularization \
--use_graceful_regularization \
--num_warmup_steps 300 \

deepspeed --master_port 30002 experiments/mistral_instruct_tuning.py \
--do_pretraining True \
--use_wandb \
--use_sparse_model \
--targeted_sparsity 0.8 \
--is_plot \
--set_sparsity_aware_threshold \
--print_sparsity \
--train_batch_size 4 \
--test_batch_size 16 \
--max_steps 10000 \
--model_name "mistral_sparse_80_percent_\"$data\"_graceful_reg_500steps " \
--use_sparse_regularization \
--use_graceful_regularization \
--num_warmup_steps 500 \

deepspeed --master_port 30002 experiments/mistral_instruct_tuning.py \
--do_pretraining True \
--use_wandb \
--use_sparse_model \
--targeted_sparsity 0.9 \
--is_plot \
--set_sparsity_aware_threshold \
--print_sparsity \
--train_batch_size 4 \
--test_batch_size 16 \
--max_steps 10000 \
--model_name "mistral_sparse_90_percent_\"$data\"_graceful_reg_300steps " \
--use_sparse_regularization \
--use_graceful_regularization \
--num_warmup_steps 300 \

deepspeed --master_port 30002 experiments/mistral_instruct_tuning.py \
--do_pretraining True \
--use_wandb \
--use_sparse_model \
--targeted_sparsity 0.9 \
--is_plot \
--set_sparsity_aware_threshold \
--print_sparsity \
--train_batch_size 4 \
--test_batch_size 16 \
--max_steps 10000 \
--model_name "mistral_sparse_90_percent_\"$data\"_graceful_reg_500steps " \
--use_sparse_regularization \
--use_graceful_regularization \
--num_warmup_steps 500 \

## Sparse Mistral Pretrained with 30K Steps
#deepspeed experiments/mistral_instruct_tuning.py --use_wandb --do_pretraining --use_sparse_model --targeted_sparsity 0.8 \
#--is_plot --set_sparsity_aware_threshold --print_sparsity --max_steps 5000 \
#--train_batch_size 2 --test_batch_size 16 --use_sparse_regularization \
#--push_to_hub True --model_name "sparse_sparse_80_percent_pretraining_warmup_500_0_1_steps_5k_distillation"