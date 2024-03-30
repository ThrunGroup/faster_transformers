import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict

import os
import json
import matplotlib.pyplot as plt

from datetime import date


def optimize_annotations(steps, values):
    # This will store the modified y positions of annotations to avoid overlap
    optimized_y = []

    # Starting with the first annotation y position
    last_y = values[0]
    for y in values:
        adjusted_y = y
        if (
            abs(adjusted_y - last_y) < 0.01
        ):  # if the difference is too small, adjust it a bit
            adjusted_y = last_y + 0.01  # Adjust the position to prevent overlap
        optimized_y.append(adjusted_y)
        last_y = adjusted_y

    return optimized_y


def plot():
    # Parent directory where all model directories are stored
    parent_directories = [
        "/scr/jay/output/2024-03-11",
        "/scr/jay/output/2024-03-12",
        "/scr/jay/output/2024-03-20",
        # "matx/u/vxbrando/output/2024-03-28",
    ]
    # parent_directories = ["/scr/jay/output/2024-03-21", "/scr/jay/output/2024-03-22",]

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
    plt.figure(figsize=(8, 6))
    for model_type, accuracies in model_accuracies.items():
        steps = list(sorted(accuracies.keys()))
        del steps[1]
        values = [accuracies[step] for step in steps]

        if "50" in model_type:
            model_type = "CATS 50%"
            color = "blue"
        elif "70" in model_type:
            model_type = "CATS 70%"
            color = "orange"

        elif "90" in model_type:
            model_type = "CATS 90%"
            color = "green"

        elif "relu" in model_type:
            model_type = "ReLU"
            color = "red"


        plt.plot(
            steps,
            values,
            label=model_type,
            marker="o",
            linewidth=2,
            color=color,
        )
        print(model_type, values)

        # optimized_values = optimize_annotations(steps, values)
        #
        # # Annotating each point with its accuracy
        # for i, (x, y, txt) in enumerate(zip(steps, optimized_values, values)):
        #     plt.annotate(
        #         f"{txt:.2f}",
        #         (x, y),
        #         textcoords="offset points",
        #         xytext=(0, 5),
        #         ha="center",
        #         fontsize=7,
        #         clip_on=True,
        #     )
        # Annotating each point with its accuracy
        if "50" in model_type:
            ytext = 5
        elif "70" in model_type:
            ytext = 0
            print(model_type)
        else:
            ytext = 5
        for i, txt in enumerate(values):
            plt.annotate(
                f"{txt:.4f}",
                (steps[i], values[i]),
                textcoords="offset points",
                xytext=(0, ytext),
                ha="center",
            )

    if base_model_accuracy > 0:
        plt.plot(
            steps,
            [base_model_accuracy for _ in steps],
            label="Mistral 7B",
            linestyle="--",
            color="gray",
            linewidth=2,
        )
        plt.annotate(
            f"{base_model_accuracy:.3f}",
            (steps[0], base_model_accuracy),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.xlabel("Steps (batch size 16)")
    plt.ylabel("Average Accuracy")
    plt.title("Average 0-shot Accuracy Comparison")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")
    plt.tight_layout()
    plt.legend()
    plt.show()

    # Save the figure
    dirname = "/sailhome/vxbrando/faster_transformers/figures"
    os.makedirs(dirname, exist_ok=True)
    plt.savefig(f"{dirname}/{date.today()}_Model Comparisons zero shot accuracy.svg")


def plot_sparsity_by_layers():
    # First model data
    layer_sparsity_mistral = {
        0: 46.871,
        1: 48.670,
        2: 48.708,
        3: 48.915,
        4: 49.830,
        5: 49.737,
        6: 50.185,
        7: 49.768,
        8: 51.200,
        9: 50.653,
        10: 50.350,
        11: 49.871,
        12: 49.788,
        13: 49.830,
        14: 48.172,
        15: 49.192,
        16: 48.596,
        17: 50.086,
        18: 49.952,
        19: 50.237,
        20: 50.838,
        21: 51.491,
        22: 51.961,
        23: 51.736,
        24: 52.297,
        25: 51.993,
        26: 52.720,
        27: 52.029,
        28: 50.980,
        29: 49.977,
        30: 49.303,
        31: 48.358,
    }
    total_sparsity_mistral = 50.134

    # Second model data
    layer_sparsity_llama2 = {
        0: 43.458,
        1: 54.273,
        2: 61.952,
        3: 59.508,
        4: 58.456,
        5: 56.079,
        6: 59.057,
        7: 57.170,
        8: 59.312,
        9: 59.845,
        10: 58.270,
        11: 59.132,
        12: 57.869,
        13: 57.456,
        14: 57.388,
        15: 57.030,
        16: 58.296,
        17: 58.626,
        18: 59.365,
        19: 62.269,
        20: 65.336,
        21: 65.934,
        22: 66.911,
        23: 66.425,
        24: 66.758,
        25: 65.723,
        26: 64.741,
        27: 64.395,
        28: 61.839,
        29: 59.105,
        30: 57.376,
        31: 57.589,
    }
    total_sparsity_llama2 = 59.905

    # Plotting both models on the same graph
    plt.figure(figsize=(14, 8))

    # Mistral model
    plt.plot(
        list(layer_sparsity_mistral.keys()),
        list(layer_sparsity_mistral.values()),
        label=f"Mistral 7b Model with 50% CATS Sparsity",
        marker="o",
        color="b"
    )

    # Llama2 model
    plt.plot(
        list(layer_sparsity_llama2.keys()),
        list(layer_sparsity_llama2.values()),
        label=f"Llama2 7b Model with 50% CATS Sparsity",
        marker="o",
        color="g"
    )

    # Adding horizontal lines for total sparsity
    plt.axhline(
        y=total_sparsity_mistral,
        color="b",
        linestyle="--",
        label=f"Total Sparsity Mistral: {total_sparsity_mistral}%",
    )
    plt.axhline(
        y=total_sparsity_llama2,
        color="g",
        linestyle="--",
        label=f"Total Sparsity Llama2: {total_sparsity_llama2}%",
    )

    # Adding labels and title
    plt.xlabel("Transformer Layer")
    plt.ylabel("% Activated Neurons")
    plt.title("Activation Sparsity in Sparse Models")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/activation_sparsity.svg")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    plot()
    # plot_sparsity_by_layers()
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
