import wandb
import argparse
from experiments.mistral_instruct_tuning import train, is_running_deepspeed
from datetime import datetime


def get_sweep_config(
    dataset_type: str = "cola",
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
):
    sweep_config = {
        "method": "random",
        "metric": {"name": "test_accuracy", "goal": "maximize"},
        "parameters": {
            "seed": {"distribution": "constant", "value": 0},
            "model_name": {"distribution": "constant", "value": "Mistral_Sparse_0.1"},
            "dataset_type": {"distribution": "constant", "value": dataset_type},
            "output_dir": {"distribution": "constant", "value": "/scr/jay/ckpt"},
            "num_epochs": {"distribution": "constant", "value": 8},
            "push_to_hub": {"distribution": "constant", "value": False},
            "model_save": {"distribution": "constant", "value": False},
            "train_batch_size": {"distribution": "constant", "value": 8},
            "test_batch_size": {"distribution": "constant", "value": 16},
            "gradient_checkpointing": {"distribution": "constant", "value": False},
            "local_rank": {"distribution": "constant", "value": 0},
            "is_debugging": {"distribution": "constant", "value": False},
            "is_plot": {"distribution": "constant", "value": False},
            "set_sparsity_aware_threshold": {"distribution": "constant", "value": True},
            "print_act_stats": {"distribution": "constant", "value": False},
            "print_sparsity": {"distribution": "constant", "value": True},
            "use_sparse_model": {"distribution": "constant", "value": True},
            "use_sparse_regularization": {
                "distribution": "categorical",
                "values": [True, False],
            },
            "targeted_sparsity": {
                "distribution": "categorical",
                "values": [0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99],
            },
        },
    }
    return sweep_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep script for MistralModel"
    )
    parser.add_argument("--project_name", type=str, default="Sweep_SparseMistralMLP")
    parser.add_argument("--dataset_type", type=str, default="cola")
    parser.add_argument("--local_rank", type=int, default=0)  # To run deepspeed
    config = parser.parse_args()
    project_name = config.project_name
    if is_running_deepspeed():
        project_name += "_deepspeed"

    now = datetime.now()
    now = now.strftime("%Y-%m-%d %Hh%Mm%Ss")
    project_name += f"_{now}"

    wandb.login()
    sweep_config = get_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project=project_name)

    def wrapper_for_train(exp_config=None):
        return train(exp_config=exp_config, use_wandb=True, use_sweep=True)

    wandb.agent(sweep_id, wrapper_for_train, count=10)
