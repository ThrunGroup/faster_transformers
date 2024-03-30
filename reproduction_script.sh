#!/bin/bash

# Install the requirements
pip install -r requirements.txt

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

