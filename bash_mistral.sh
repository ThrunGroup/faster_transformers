#!/bin/bash


accelerate launch experiments/mistral_downstream.py --exp 1
accelerate launch experiments/mistral_downstream.py --exp 2
accelerate launch experiments/mistral_downstream.py --exp 3