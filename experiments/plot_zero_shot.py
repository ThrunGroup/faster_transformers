import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict

import os
import json
import matplotlib.pyplot as plt

from datetime import date


def plot(
    parent_directories,
    base_model_name,
    sparse_model_name,
    relu_model_name,
):
    # List to store directory names
    directories = []

    # Automatically find directories that match the naming convention
    for parent_directory in parent_directories:
        for entry in os.scandir(parent_directory):
            if entry.is_dir() and any(
                model_name in entry.name for model_name in [base_model_name, sparse_model_name, relu_model_name]
            ):
                directories.append(entry.path)
    print("directories: ", directories)

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

                if base_model_name in directory:
                    base_model_accuracy = avg_acc
                    print("Base model acc: ", base_model_accuracy)
                else:
                    # Extract step number and model type from directory name
                    parts = directory.split("_")
                    if relu_model_name in directory:
                        model_type = "relu"
                    else:
                        model_type = parts[3]  # Adjusted for the full path
                    step = int(parts[-1].replace("steps", ""))

                    if model_type not in model_accuracies:
                        model_accuracies[model_type] = {}
                    model_accuracies[model_type][step] = avg_acc

    # print(model_accuracies)
    # return

    # Plotting the results
    plt.figure(figsize=(10, 6))
    for model_type, accuracies in model_accuracies.items():
        steps = sorted(accuracies.keys())
        values = [accuracies[step] for step in steps]
        plt.plot(steps, values, label=model_type, marker="o")
        for step in steps:
            print("model type: ", model_type, "step: ", step, "accuracy: ", accuracies[step])

    if base_model_accuracy > 0:
        plt.plot(steps, [base_model_accuracy for _ in steps], label="Base Mistral")

    plt.xlabel("Steps (batch size 16)")
    plt.ylabel("Average Accuracy")
    plt.title("Average 0-shot Accuracy Comparison")
    plt.legend()
    plt.show()

    # Save the figure
    dirname = "/matx/u/lukeai/sparse_silu/figures"
    os.makedirs(dirname, exist_ok=True)
    plt.savefig(f"{dirname}/{date.today()}_Model Comparisons zero shot accuracy.png")
    print(f"{dirname}/{date.today()}_Model Comparisons zero shot accuracy.png")


if __name__ == "__main__":
    parent_directories = [
        "/matx/u/lukeai/sparse_silu/output",
    ]
    prefixes = [
        "llama-2-7b-chat-hf",
    ]
    plot(
        parent_directories,
        base_model_name="llama-2-7b-chat-hf",
        sparse_model_name="llama_sparse",
        relu_model_name="llama_relu",
    )
