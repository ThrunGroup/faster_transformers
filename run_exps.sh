#!/bin/bash

python experiments/mistral_instruct_tuning.py --is_debugging True --set_sparsity_aware_threshold True \
--print_act_stats True --is_plot True
