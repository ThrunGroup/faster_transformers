#!/bin/bash

source /sailhome/vxbrando/.bashrc
conda activate mistral

accelerate launch -m --main_process_port 31212 lm_eval --model hf \
    --model_args pretrained=ckpt/Mistral_Sparse_refined_web_50p_graceful_True,trust_remote_code=True \
    --tasks arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --batch_size auto \
    --log_samples \
    --output_path output/Mistral_Sparse_refined_web_50p_100steps_graceful_True
