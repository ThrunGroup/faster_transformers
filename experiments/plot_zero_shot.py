import os
import json
import matplotlib.pyplot as plt
import re


def get_mean_accuracy(parent_prefix, directory):
    path = os.path.join(parent_prefix, directory, "results.json")
    if os.path.exists(path):
        with open(path, "r") as file:
            data = json.load(file)
            acc_sum = 0
            count = 0
            # print("\n\n---directory---", directory)
            for key, value in data["results"].items():
                # print(key, value["acc,none"])
                acc_sum += value["acc,none"]
                count += 1
            avg_acc = acc_sum / count if count > 0 else 0
            return avg_acc
    return 0


def plot(
    model_name,
    parent_directories,
    base_model_name,
    sparse_model_name,
    relu_model_name,
):
    directories = []

    for parent_directory in parent_directories:
        for entry in os.scandir(parent_directory):
            if entry.is_dir() and any(
                # model_name in entry.name for model_name in [base_model_name, sparse_model_name, relu_model_name]
                re.match(model_name, entry.name)
                for model_name in [base_model_name, sparse_model_name, relu_model_name]
            ):
                directories.append(entry.path)

    # print("Directories:", directories)
    model_accuracies = {}
    base_model_accuracy = 0

    for directory in directories:
        if os.path.exists(os.path.join(directory, "results.json")):
            avg_acc = get_mean_accuracy(parent_directories[0], directory)

            if base_model_name in directory:
                base_model_accuracy = avg_acc
                # print("Base model accuracy:", base_model_accuracy)
            else:
                parts = directory.split("_")
                if "relu" in directory:
                    model_type = "ReLUfication"
                else:
                    sparsity = parts[3].split("p")[0]
                    model_type = f"CATS {sparsity}%"
                step = int(parts[-1].replace("steps2", "").replace("steps", ""))
                if step == 100:
                    continue
                model_accuracies[model_type] = model_accuracies.get(model_type, {})
                model_accuracies[model_type][step] = avg_acc

    plt.figure(figsize=(8, 6))

    model_types = list(sorted(model_accuracies.keys()))

    for model_type in model_types:
        accuracies = model_accuracies[model_type]
        steps = sorted(accuracies.keys())
        values = [accuracies[step] for step in steps]
        color = "#227CF6"
        alpha = 1.0
        xytext = (0, 10)
        if "50" in model_type:
            xytext = (0, 14)
        elif "70" in model_type:
            alpha = 0.6
            xytext = (0, -14)
        elif "90" in model_type:
            alpha = 0.3
            xytext = (0, -14)
        elif "ReLUfication" == model_type:
            alpha = 1.0
            color = "#FC778D"

        plt.plot(steps, values, label=model_type.capitalize(), marker="o", linewidth=2, color=color, alpha=alpha)
        for step, accuracy in accuracies.items():
            if int(step) == 1 and "50" in model_type:
                print(model_type)
                xytext = (0, -10)
            plt.annotate(f"{accuracy:.4f}", (step, accuracy), textcoords="offset points", xytext=xytext, ha="center")

    if base_model_accuracy > 0:
        plt.plot(
            steps, [base_model_accuracy] * len(steps), label="Mistral 7B", linestyle="--", color="gray", linewidth=2
        )
        plt.annotate(
            f"{base_model_accuracy:.4f}",
            (steps[0], base_model_accuracy),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
        )
    plt.xlabel("Finetuning Steps (batch size 16)")
    plt.ylabel("Average 0-Shot Accuracy")
    plt.legend(fontsize=12, labels=model_types + ["Mistral 7B"])
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")
    plt.tight_layout()

    dirname = "/matx/u/lukeai/sparse_silu/figures"
    os.makedirs(dirname, exist_ok=True)
    plt.savefig(f"{dirname}/{model_name}_training_steps_vs_zero_shot_accuracy.png", dpi=300)
    print(f"{dirname}/{model_name}_training_steps_vs_zero_shot_accuracy.png")


if __name__ == "__main__":
    # LLAMA
    # parent_directories = [
    #     "/matx/u/lukeai/sparse_silu/output",
    # ]
    # plot(
    #     "llama",
    #     parent_directories,
    #     base_model_name="llama_7b_hf",
    #     sparse_model_name="llama_sparse.*2",
    #     relu_model_name="relu_llama.*",
    # )

    # MISTRAL
    parent_directories = [
        "/scr/jay/output/2024-03-11",
        "/scr/jay/output/2024-03-12",
        "/scr/jay/output/2024-03-20",
    ]
    plot(
        "mistral",
        parent_directories,
        base_model_name="mistral_base",
        sparse_model_name="mistral_sparse_refined_.*",
        relu_model_name="mistral_sparse_refined_relu_.*",
    )
    # directories = [
    #     "llama_7b_hf",
    #     "llama_sparse_50p_1steps2",
    #     "llama_sparse_70p_1steps2",
    #     "llama_sparse_90p_1steps2",
    #     "llama_sparse_70p_1steps2",
    #     "relu_llama_1steps",
    # ]
    # for directory in directories:
    #     print(directory, get_mean_accuracy(parent_directories[0], directory))
