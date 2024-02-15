#!/bin/bash

source /sailhome/vxbrando/.bashrc
conda activate mistral

accelerate launch -m --main_process_port 31212 lm_eval --model hf \
    --model_args pretrained=/scr/jay/ckpt/Mistral_Sparse_refined_web_50p,trust_remote_code=True \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --batch_size auto \
    --log_samples \
    --output_path output/mistral_sparse_refined_50p_750steps




#accelerate launch -m --main_process_port 31212 lm_eval --model hf \
#    --model_args pretrained=/scr/jay/ckpt/Mistral_Sparse_refined_web_50p,trust_remote_code=True \
#    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
#    --batch_size auto \
#    --log_samples \
#    --output_path output/mistral_sparse_refined_50p_1500steps

#
#
#accelerate launch -m --main_process_port 31212 lm_eval --model hf \
#    --model_args pretrained=/scr/jay/ckpt/Mistral_Sparse_refined_web_70,trust_remote_code=True\
#    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,mmlu\
#    --batch_size 16 \
#    --log_samples \
#    --output_path output/mistral_sparse_0.7

#accelerate launch -m --main_process_port 31212 lm_eval --model hf \
#    --model_args pretrained=/scr/jay/ckpt/Mistral_Sparse_refined_web_70p,trust_remote_code=True\
#    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq\
#    --batch_size auto:4 \
#    --log_samples \
#    --output_path output/mistral_sparse_0.7
#accelerate launch -m --main_process_port 31212 lm_eval --model hf \
#    --model_args pretrained=/scr/jay/ckpt/Mistral_Sparse_refined_web_70percent,trust_remote_code=True\
#    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq\
#    --batch_size auto:4 \
#    --log_samples \
#    --output_path output/mistral_sparse_0.7

#accelerate launch -m --main_process_port 31212 lm_eval --model hf \
#    --model_args pretrained=/scr/jay/ckpt/Mistral_Sparse_refined_web_70,trust_remote_code=True\
#    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq\
#    --batch_size auto:4 \
#    --log_samples \
#    --output_path output/mistral_sparse_0.7

#accelerate launch -m --main_process_port 31212 lm_eval --model hf \
#    --model_args pretrained=/scr/jay/ckpt/sparse_models/unit_test,trust_remote_code=True\
#    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq\
#    --batch_size auto\
#    --log_samples \
#    --output_path output/mistral_sparse_0.7

# /scr/jay/ckpt/Mistral_Sparse_refined_web_70