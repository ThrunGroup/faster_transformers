#!/bin/bash

accelerate launch -m --main_process_port 12329 lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-chat-hf,trust_remote_code=True \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
    --batch_size 16 \
    --log_samples \
    --output_path output/llama-2-7b-chat-hf

