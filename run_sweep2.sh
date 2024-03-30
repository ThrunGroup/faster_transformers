#!/bin/bash
source /sailhome/vxbrando/.bashrc
conda activate mistral

#deepspeed experiments/hyperparam_sweep.py --dataset_type cola --train_batch_size 4 --test_batch_size 4
#deepspeed experiments/hyperparam_sweep.py --dataset_type boolq --train_batch_size 4 --test_batch_size 4

bash pretrain_s09.sh
bash pretrain_relu.sh
