#!/bin/bash


deepspeed  experiments/create_models.py --exp 1
#deepspeed experiments/create_models.py --exp 2
#deepspeed  experiments/mistral_downstream.py --exp 3