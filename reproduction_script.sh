#!/bin/bash

# Install the requirements
pip install -e .
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
# Install the flash_gemv package
cd flash_gemv
pip install -e .
cd ..
echo "Hello"

# Get the root folder for faster_transformer
export PYTHONPATH=${PYTHONPATH}:$PWD
project_path=$PWD
act_path=$1
ckpt_path=$2
result_path=$3
export CATS_ACTPATH=$act_path
export CATS_CKPTPATH=$ckpt_path
export CATS_RESPATH=$result_path

# Collect the statistics before General finetuning
bash scripts/plot_mlp_histogram.sh $ckpt_path
bash scripts/zero_shot_evaluation_without_general_finetuning.sh $ckpt_path
bash scripts/evaluate_base_model.sh

## Run general finetuning
bash scripts/general_finetuning_cats.sh $ckpt_path
bash scripts/general_finetuning_llama_relufication.sh $ckpt_path
##
bash scripts/general_finetuning_mistral_cats.sh $ckpt_path
bash scripts/general_finetuning_mistral_relufication.sh $ckpt_path

## Plot activation sparsity after general finetuning (saved in "figures" directory)
bash scripts/plot_post_training_activation_sparsity_per_layer.sh $ckpt_path

## Benchmark MLP Block
cd flash_gemv/bench/
bash final_profile_llama7B.sh
bash final_profile_mistral7B.sh
python plot_latency_sub.py
cd $project_path

# Benchmark Generation
bash scripts/bench_generation_llama7B.sh $project_path $act_path $ckpt_path
bash scripts/bench_generation_mistral7B.sh $project_path $act_path $ckpt_path
