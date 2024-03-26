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
    parent_directories = [
        "/matx/u/lukeai/sparse_silu/output",
    ]

    # List to store directory names
    directories = []

    # Automatically find directories that match the naming convention
    for parent_directory in parent_directories:
        for entry in os.scandir(parent_directory):
            print(entry.name, entry.is_dir())
            if entry.is_dir() and entry.name.startswith("llama-2-7b-chat-hf"):
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

                if "llama-2-7b-chat-hf" in directory:
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
    dirname = "/matx/u/lukeai/sparse_silu/figures"
    os.makedirs(dirname, exist_ok=True)
    plt.savefig(f"{dirname}/{date.today()}_Model Comparisons zero shot accuracy.png")


plot()


if __name__ == "__main__":
    plot()
