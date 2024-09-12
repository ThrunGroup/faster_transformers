accelerate launch cats_exp/sample_cats_exp.py

accelerate launch -m --main_process_port 12329 lm_eval --model hf \
        --model_args pretrained=cats_exp/mistral_cats_zero_shot,trust_remote_code=True \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,sciq \
        --batch_size 16 \
        --log_samples \
        --output_path output/cats_mistral