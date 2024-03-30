#!/bin/bash

# Install the requirements
pip install -r requirements.txt

# Install the flash_gemv package
cd flash_gemv
pip install -e .
cd ..

# Get the root folder for faster_transformer
export PYTHONPATH=${PYTHONPATH}:$PWD
project_path=$PWD
act_path=$1
ckpt_path=$2
result_path=$3
export CATS_ACTPATH=$act_path
export CATS_CKPTPATH=$ckpt_path
export CATS_RESPATH=$result_path

cd scripts
# Collect the statistics before General finetuning
bash plot_mlp_histogram.sh $ckpt_path
bash zero_shot_evaluation_without_general_finetuning.sh $ckpt_path
bash evaluate_base_model.sh

# Run general finetuning
bash general_finetuning_cats.sh $ckpt_path
bash general_finetuning_llama_relufication.sh $ckpt_path

bash general_finetuning_mistral_cats.sh $ckpt_path
bash general_finetuning_mistral_relufication.sh $ckpt_path

# Plot activation sparsity after general finetuning
bash plot_post_training_activation_sparsity_per_layer.sh $ckpt_path

# Benchmark MLP Block
cd flash_gemv/bench/
bash final_profile_llama7B.sh
bash final_profile_mistral7B.sh
cd $project_path

# Benchmark Generation
bash scripts/bench_generation_llama7B.sh $project_path $act_path $ckpt_path
bash scripts/bench_generation_mistral7B.sh $project_path $act_path $ckpt_path
