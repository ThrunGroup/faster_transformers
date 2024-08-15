import matplotlib.pyplot as plt
import json
import re
import os

from utils.utils import plot_post_training_sp


def plot_sparsities():
    for target_model_type in ["llama", "mistral"]:
        log_directory = "exp_logs"
        cats_50p = {}
        cats_70p = {}
        cats_90p = {}
        for filename in os.listdir(log_directory):
            if filename.endswith(".json"):

                # Extract the percentage value
                percentage_match = re.search(r"(\d+)p", filename)
                percentage = percentage_match.group(1) if percentage_match else None

                # Extract the model name
                model_match = re.search(r"(llama|mistral)", filename)
                model_name = model_match.group(1) if model_match else None

                if model_name != target_model_type:
                    continue

                with open(os.path.join(log_directory, filename), "r") as file:
                    data = json.load(file)

                if int(percentage) == 50:
                    cats_50p = data["sparsity_list"]
                elif int(percentage) == 70:
                    cats_70p = data["sparsity_list"]
                elif int(percentage) == 90:
                    cats_90p = data["sparsity_list"]

        if target_model_type == "mistral":
            title = "Mistral-7B"
        elif target_model_type == "llama":
            title = "Llama2-7B"
        else:
            raise NotImplementedError(f"{target_model_type} isn't implemented")
        plot_post_training_sp(cats_50p, cats_70p, cats_90p, title)



    assert len(cats_50p) == len(cats_70p) == len(cats_90p), "All experiments should have been run at 50%, 70%, and 90% sparsity level"


if __name__ == "__main__":
    plot_sparsities()