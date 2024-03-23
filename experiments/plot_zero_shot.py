import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict

import os
import json
import matplotlib.pyplot as plt

from datetime import date

def plot():
    # Parent directory where all model directories are stored
    # parent_directories = ["/scr/jay/output/2024-03-11", "/scr/jay/output/2024-03-12", "/scr/jay/output/2024-03-20"]
    parent_directories = ["/scr/jay/output/2024-03-21", "/scr/jay/output/2024-03-22",]

    # List to store directory names
    directories = []

    # Automatically find directories that match the naming convention
    for parent_directory in parent_directories:
        for entry in os.scandir(parent_directory):
            if entry.is_dir() and entry.name.startswith("mistral"):
                directories.append(entry.path)
    print(directories)

    # Initialize dictionary to store accuracies
    model_accuracies = {}
    base_model_accuracy = 0

    # Loop through directories to read and process each results.json
    for directory in directories:
        if os.path.exists(os.path.join(directory, "results.json")):
            with open(os.path.join(directory, "results.json"), "r") as file:
                data = json.load(file)
                acc_sum = 0
                count = 0
                for key, value in data["results"].items():
                    acc_sum += value["acc,none"]
                    count += 1
                avg_acc = acc_sum / count if count > 0 else 0

                if "mistral_base" in directory:
                    base_model_accuracy = avg_acc
                    print(base_model_accuracy)
                    print(directory)
                else:
                    # Extract step number and model type from directory name
                    parts = directory.split("_")
                    step = int(parts[-1].replace("steps", ""))
                    model_type = parts[3]  # Adjusted for the full path

                    if model_type not in model_accuracies:
                        model_accuracies[model_type] = {}
                    model_accuracies[model_type][step] = avg_acc

    # Plotting the results
    plt.figure(figsize=(10, 6))
    for model_type, accuracies in model_accuracies.items():
        steps = sorted(accuracies.keys())
        values = [accuracies[step] for step in steps]
        plt.plot(steps, values, label=model_type, marker="o")
        print(model_type, values)

    if base_model_accuracy > 0:
        plt.plot(steps, [base_model_accuracy for _ in steps], label="Base Mistral")

    plt.xlabel("Steps (batch size 16)")
    plt.ylabel("Average Accuracy")
    plt.title("Average 0-shot Accuracy Comparison")
    plt.legend()
    plt.show()

    # Save the figure
    dirname = "/sailhome/vxbrando/faster_transformers/figures"
    os.makedirs(dirname, exist_ok=True)
    plt.savefig(f"{dirname}/{date.today()}_Model Comparisons zero shot accuracy.png")


plot()


if __name__ == "__main__":
    plot()
# def plot():
#     # Directory containing the JSON files
#     directory = "/scr/jay/output/2024-03-11/"
#
#
#
#     # Loop over the files in the directory
#     for dirname in os.listdir(directory):
#         parts = dirname.split("refined_")
#         parts = parts[1].split("_")
#         sparsify_type = parts[0]
#         steps = parts[1]
#         for filename in os.listdir(dirname):
#             if filename.endswith("results.json"):
#                 # Construct the full file path
#                 file_path = os.path.join(directory, dirname, filename)
#
#                 # Load the JSON data
#                 with open(file_path, "r") as f:
#                     data = json.load(f)
#
#                 # Store the data
#                 results = data["results"]
#                 datasets = list(results.keys())
#                 avg_accuracy = 0
#                 counts = 0
#                 for dataset in datasets:
#                     if "acc_none" in results[dataset]:
#                         avg_accuracy += results[dataset]["acc,none"]
#                     counts += 1
#                 avg_accuracy /= counts
#
#         # Plotting
#
#         batch_sizes = sorted(
#             set(
#                 bs
#                 for l2_dict in data_dict.values()
#                 for bs_dict in l2_dict.values()
#                 for bs in bs_dict.keys()
#             )
#         )
#         color_map = plt.cm.get_cmap("viridis", len(batch_sizes))
#         batch_size_colors = {bs: color_map(i) for i, bs in enumerate(batch_sizes)}
#
#         for hidden, l2_dict in data_dict.items():
#             for l2, bs_dict in l2_dict.items():
#                 if hidden == "790":
#                     plt.figure(figsize=(14, 6))
#
#                     # Error accuracy subplot
#                     plt.subplot(1, 2, 1)
#                     for bs in sorted(bs_dict.keys(), key=int):  # Sort batch sizes
#                         values = bs_dict[bs]
#                         # Compute error accuracy as 1 - eval_accuracy for each point
#                         error_accuracy = [1 - acc for acc in values["eval_accuracy"]]
#                         plt.plot(
#                             error_accuracy,
#                             label=f"BS: {bs}",
#                             color=batch_size_colors[bs],
#                         )
#                     plt.title(f"Error Accuracy vs Epoch (Hidden: {hidden}, L2: {l2})")
#                     plt.xlabel("Epoch")
#                     plt.ylabel("Error Accuracy (log scale)")
#                     plt.yscale("log")
#                     plt.legend()
#
#                     # Sparsity subplot
#                     plt.subplot(1, 2, 2)
#                     for bs in sorted(bs_dict.keys(), key=int):  # Sort batch sizes
#                         values = bs_dict[bs]
#                         plt.plot(
#                             values["sparsity"],
#                             label=f"BS: {bs}",
#                             color=batch_size_colors[bs],
#                         )
#                     plt.title(f"Sparsity vs Epoch (Hidden: {hidden}, L2: {l2})")
#                     plt.xlabel("Epoch")
#                     plt.ylabel("Sparsity")
#                     plt.legend()
#
#                     plt.tight_layout()
#                     plt.show()
