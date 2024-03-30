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

# Collect the statistics before General finetuning
bash plot_mlp_histogram.sh
bash zero_shot_evaluation_without_general_finetuning.sh
bash scripts/evaluate_base_model.sh

# Run general finetuning
bash general_finetuning_cats.sh
bash general_finetuning_llama_relufication.sh

bash general_finetuning_mistral_cats.sh
bash general_finetuning_mistral_relufication.sh

# Plot activation sparsity after general finetuning
bash plot_post_training_activation_sparsity_per_layer.sh

# Benchmark MLP Block
bash flash_gemv/bench/final_profile_llama7B.sh
bash flash_gemv/bench/final_profile_mistral7B.sh

# Benchmark Generation
bash scripts/bench_generation_llama7B.sh $project_path
bash scripts/bench_generation_mistral7B.sh $project_path
