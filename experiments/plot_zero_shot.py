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
    directories = []

    for parent_directory in parent_directories:
        for entry in os.scandir(parent_directory):
            if entry.is_dir() and any(
                model_name in entry.name for model_name in [base_model_name, sparse_model_name, relu_model_name]
            ):
                directories.append(entry.path)
    print("Directories:", directories)

    model_accuracies = {}
    base_model_accuracy = 0

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
                    print("Base model accuracy:", base_model_accuracy)
                else:
                    parts = directory.split("_")
                    model_type = "relu" if relu_model_name in directory else parts[3]
                    step = int(parts[-1].replace("steps", ""))
                    model_accuracies[model_type] = model_accuracies.get(model_type, {})
                    model_accuracies[model_type][step] = avg_acc

    plt.figure(figsize=(8, 6))
    for model_type, accuracies in model_accuracies.items():
        steps = sorted(accuracies.keys())
        values = [accuracies[step] for step in steps]
        plt.plot(steps, values, label=model_type.capitalize(), marker="o", linewidth=2)
        for step, accuracy in accuracies.items():
            plt.annotate(f"{accuracy:.4f}", (step, accuracy), textcoords="offset points", xytext=(0, 10), ha="center")

    if base_model_accuracy > 0:
        plt.plot(
            steps, [base_model_accuracy] * len(steps), label="Llama2 7B", linestyle="--", color="gray", linewidth=2
        )
        plt.annotate(
            f"{base_model_accuracy:.4f}",
            (steps[0], base_model_accuracy),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
    plt.xlabel("Finetuning Steps (batch size 4)", fontsize=14)
    plt.ylabel("Average 0-Shot Accuracy", fontsize=14)
    # plt.title("Average 0-Shot Accuracy", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")
    plt.tight_layout()

    dirname = "/matx/u/lukeai/sparse_silu/figures"
    os.makedirs(dirname, exist_ok=True)
    plt.savefig(f"{dirname}/llama_training_steps_vs_zero shot_accuracy", dpi=300)
    plt.show()
    print(f"Figure saved: {dirname}/Model_Comparisons_zero_shot_accuracy_Llama.png")


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
