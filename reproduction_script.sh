#!/bin/bash

# Install the requirements
pip install -e .
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

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
export PROJECT_PATH=$project_path
export CATS_ACTPATH="$project_path/$act_path"
export CATS_CKPTPATH="$project_path/$ckpt_path"
export CATS_RESPATH="$project_path/$result_path"

### Collect the statistics before General finetuning & plot figure 1
#bash scripts/plot_mlp_histogram.sh $ckpt_path

### Plot figure 5: activation sparsity (figure 5) after general finetuning
#bash scripts/plot_post_training_act_sparsity.sh $ckpt_path

### Experiments for Table 1. Results are saved in "output/"
#bash scripts/zero_shot_evaluation_without_general_finetuning.sh $ckpt_path

### Experiments for Table 2 and 3 (Relevant data will be stored in exp_logs)
#bash scripts/instruction_tuning.sh

### Run general finetuning
bash scripts/general_finetuning_cats.sh $ckpt_path
#bash scripts/general_finetuning_llama_relufication.sh $ckpt_path

bash scripts/general_finetuning_mistral_cats.sh $ckpt_path
bash scripts/general_finetuning_mistral_relufication.sh $ckpt_path
python experiments/plot_zero_shot.py # Plot figure 1



### Benchmark MLP Block
cd flash_gemv/bench/
bash final_profile_llama7B.sh
bash final_profile_mistral7B.sh
# Plot fig 3 and fig 6
python plot_latency_sub.py

# Benchmark Generation (figure 4)
cd $project_path
bash scripts/bench_generation_llama7B.sh $project_path $act_path $ckpt_path
cd $project_path
bash scripts/bench_generation_mistral7B.sh $project_path $act_path $ckpt_path
