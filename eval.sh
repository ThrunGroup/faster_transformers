#!/bin/bash

source /sailhome/vxbrando/.bashrc
conda activate mistral

date=$(date +%Y-%m-%d)

#lm_eval --model hf \
#        --model_args pretrained=/scr/jay/ckpt/Mistral_Sparse_refined_web_relu,trust_remote_code=True \
#        --tasks mmlu \
#        --batch_size 16 \
#        --log_samples \
#        --output_path output/mistral_sparse_refined_relu_1000steps

#accelerate launch -m --main_process_port 14211 lm_eval --model hf \
#        --model_args pretrained=/scr/jay/ckpt/Mistral_Sparse_refined_web_50p,trust_remote_code=True \
#        --tasks mmlu \
#        --batch_size 16 \
#        --log_samples \
#        --output_path output/mistral_sparse_refined_50p_1000steps
#
#accelerate launch -m --main_process_port 14121 lm_eval --model hf \
#        --model_args pretrained=/scr/jay/ckpt/Mistral_Sparse_refined_web_50p,trust_remote_code=True \
#        --tasks mmlu \
#        --batch_size 16 \
#        --log_samples \
#        --output_path output/mistral_base
#current_steps=4750
#
date=2024-03-10
#accelerate launch -m --main_process_port 12327 lm_eval --model hf \
#    --model_args pretrained=/scr/jay/ckpt/$date/Mistral_Sparse_refined_web_relu,trust_remote_code=True \
#    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
#    --batch_size auto \
#    --log_samples \
#    --output_path /scr/jay/output/$date/mistral_sparse_refined_relu_${current_steps}steps

#accelerate launch -m --main_process_port 14121 lm_eval --model hf \
#    --model_args pretrained=mistralai/Mistral-7B-v0.1,trust_remote_code=True \
#    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
#    --batch_size auto \
#    --log_samples \
#    --output_path /scr/jay/$date/output/mistral_base


#Arc-E Arc-C Hellaswag BoolQ PIQA LAMBADA TriviaQA WinoGrande SciQ


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

accelerate launch -m --main_process_port 31212 lm_eval --model hf \
    --model_args pretrained=/scr/jay/ckpt/$date/Mistral_Sparse_refined_web_50p_no_adapter,trust_remote_code=True\
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq\
    --batch_size auto \
    --log_samples \
    --output_path /scr/jay/output/$date/mistral_sparse_refined_50p_300steps

accelerate launch -m --main_process_port 31212 lm_eval --model hf \
    --model_args pretrained=/scr/jay/ckpt/$date/Mistral_Sparse_refined_web_relu_no_adapter,trust_remote_code=True\
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq\
    --batch_size auto \
    --log_samples \
    --output_path /scr/jay/output/$date/mistral_sparse_refined_relu_300steps

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
#    --output_path /scr/jay/output/mistral_sparse_0.7


# /scr/jay/ckpt/Mistral_Sparse_refined_web_70