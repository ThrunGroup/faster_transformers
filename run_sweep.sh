#!/bin/bash

deepspeed experiments/hyperparam_sweep.py --dataset_type cola --train_batch_size 4 --test_batch_size 4
deepspeed experiments/hyperparam_sweep.py --dataset_type boolq --train_batch_size 4 --test_batch_size 4