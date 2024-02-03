import os
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoTokenizer,
)
from torch.nn import functional as F
import wandb
import gc
import torch
import argparse
import numpy as np

from experiments.data.get_dataset import get_dataset
from experiments.models.build_model import build_model
from experiments.models.sparse_mistral.relufication import (
    print_dead_neuron_stats,
)
from experiments.models.sparse_mistral.relufication import (
    SparseSiLUTrainer,
    enable_sparse_silu,
)

from utils.constants import (
    # Models
    T5,
    T5_LARGE,
    T5_3B,
    MISTRAL_7B,
    # Datasets
    COLA,
    OPENWEBTEXT,
)


def finetune(
    model_type: str,
    model_name: str,
    model,
    tokenizer,
    dataset_name: str,
    num_epochs: int = 10,
    resume_from_checkpoint: bool = False,
    run_name: str = "",
    do_evaluation: bool = True,
    push_to_hub: bool = True,
    config=None,
    save_stats: bool = False,
    use_mare_mlp: bool = False,
    use_collect_stats: bool = False,
    use_sparse_regularization: bool = False,
    use_sparse_silu: bool = False,
    use_lora: bool = True,
):
    """
    Finetune the model on a dataset

    :param model_name: Name of the model to finetune
    :param model: PyTorch model to finetune
    :param tokenizer: Tokenizer to tokenize the dataset
    :param dataset_name: Name of the dataset to train on
    :param num_epochs: Number of epochs to train the model
    :param resume_from_checkpoint: Whether to resume the training from the checkpoint
    :param do_evaluation: Whether to evaluate the model after training and save the metrics
    :param save_stats: Whether to save statistics of t5_sparsity_check model
    :returns: Dictionary of evaluation metrics and inference time if `do_evaluation` is true
    """
    print("Finetuning started...")
    model_name = model_name.split("/")[-1]
    dataset = get_dataset(dataset_name, tokenizer, model_type)
    train_dataset, val_dataset, test_dataset = dataset.get_tokenized_dataset()

    data_collator = dataset.get_data_collator()
    compute_metrics = dataset.get_compute_metrics()
    preprocess_logits_for_metrics = dataset.preprocess_logits_for_metrics

    if "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) == 0:
        wandb_run = wandb.init(
            project=f"{model_name}_{dataset_type}",
            name=model_name,
            reinit=True,
            config=config,
        )

    # @TODO: Unification of output dir

    training_args = TrainingArguments(
        output_dir=os.path.join("ckpt", model_name),
        max_steps=40,
        evaluation_strategy="steps",
        eval_steps=20,  # early stopping counts only when eval step and save step match
        save_steps=20,
        logging_steps=5,
        save_strategy="steps",
        learning_rate=1e-5,
        weight_decay=0.01,
        num_train_epochs=num_epochs,
        logging_dir="exp_logs",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=model_name,
        gradient_accumulation_steps=2,
        deepspeed="ds_config.json",
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["test_batch_size"],
        gradient_checkpointing=config["gradient_checkpointing"],
        bf16=True,
        ddp_find_unused_parameters=True,
        hub_model_id=f"thrunlab/{model_name}",
        push_to_hub=push_to_hub,
        seed=2,
        data_seed=1,
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

    trainer = SparseSiLUTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping],
        # compute_metrics=compute_metrics,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        use_sparse_regularization=use_sparse_regularization,
    )

    if use_sparse_silu:
        print(
            "Sparse SiLU enabled. It will zero out swish outputs smaller than 0.1"
        )
        enable_sparse_silu(model)

    print("Training... ")
    trainer.train()

    print("Evaluating... ")
    results = trainer.evaluate(test_dataset)
    print("Test set evaluation: ", results)

    if use_sparse_silu:
        print("Printing dead neurons")
        if use_lora:
            print_dead_neuron_stats(model.get_base_model())
        else:
            print_dead_neuron_stats(model)

    if "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) == 0:
        wandb_run = wandb.init(
            project=f"{model_name}_{dataset_type}",
            name=model_name,
            reinit=True,
            config=config,
        )

    if "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) == 0:
        wandb_run.finish()
        if push_to_hub:
            print("Pushing to hub")
            trainer.push_to_hub()
            print(f"Pushed to hub: thrunlab/{model_name}")

            os.makedirs("thrunlab", exist_ok=True)
            os.makedirs("scores", exist_ok=True)
            score_model = model.score
            torch.save(score_model.state_dict(), f"scores/{model_name}.pt")

    # Empty out gpu memory
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def create_model(
    model_type: str,
    dataset_name: str,
    num_epochs: int = 5,
    run_name: str = "",
    capacity_factor: int = 0,
    use_mare_mlp: bool = False,
    use_collect_stats: bool = False,
    num_generalists: int = 0,
    num_experts: int = 0,
    expert_size: int = 0,
    do_evaluation: bool = True,
    train_batch_size: int = 0,
    test_batch_size: int = 0,
    gradient_checkpointing: bool = False,
    generalist_ratios: float = 0,
    pretrained_model_name: str = None,
    save_stats: bool = False,
    use_sparse_silu: bool = False,
    use_sparse_regularization: bool = False,
    use_lora: bool = True,
    **kwargs,
):
    """
    Load a naive pretrained model from HuggingFace, apply the accelerator to FC layers,
    and finetune it on the specified dataset.

    :param model_type: Type of the model (TXL, GPT, etc.)
    :param dataset_name: Name of the dataset to train on
    :param num_epochs: Number of epochs to finetune the model
    :param do_evaluation: Whether to evaluate the model after training and save the metrics
    :param save_stats: Whether to save statistics of model (should set to be true only for t5_sparsity_check model)
    :return: Dictionary of evaluation logs if `do_evaluation` is true
    """
    # Build a Gated Attention model
    model_name = f"{model_type}_{dataset_name}"
    model_name += run_name

    # captures the method parameters and model_name as a dictionary
    config = locals()

    model = build_model(
        model_type,
        use_mare_mlp=use_mare_mlp or use_collect_stats,
        num_generalists=num_generalists,
        num_experts=num_experts,
        expert_size=expert_size,
        capacity_factor=capacity_factor,
        use_sparse_silu=use_sparse_silu,
        pretrained_model_name=pretrained_model_name,
    )

    model.config.pad_token_id = model.config.eos_token_id

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    finetune(
        model_type,
        model_name,
        model,
        tokenizer,
        dataset_name,
        num_epochs,
        resume_from_checkpoint=False,
        run_name=run_name,
        config=config,
        save_stats=save_stats,
        use_mare_mlp=use_mare_mlp,
        use_collect_stats=use_collect_stats,
        push_to_hub=kwargs["push_to_hub"],
        use_sparse_regularization=use_sparse_regularization,
        use_sparse_silu=use_sparse_silu,
        use_lora=use_lora,
    )


if __name__ == "__main__":
    dataset_types = [
        # SST2,
        # COLA,
        OPENWEBTEXT,
        # RTE,
        # MRPC,
        # QNLI,
        # BOOLQ,
        # MULTIRC,
        # WIC,
    ]

    model_types = [MISTRAL_7B]
    for i in range(1):
        print("EXPERIMENT: ", i)
        for model_type in model_types:
            for dataset_type in dataset_types:
                common_args = {
                    "model_type": model_type,
                    "dataset_name": dataset_type,
                    "num_epochs": 20,
                    "train_batch_size": 4,
                    "test_batch_size": 8,
                    "gradient_checkpointing": False,
                    "use_lora": True,
                }

                # Original model
                # create_model(
                #     **common_args,
                #     push_to_hub=False,  # whether to save the model both in local and remote repositories
                #     use_sparse_silu=False,  # when turned on, it kills negligible swish outputs < 0.1
                #     use_sparse_regularization=False,  # for sparse silu regularization
                #     run_name="_original_pretraining",
                # )

                # Original model
                create_model(
                    **common_args,
                    push_to_hub=False,  # whether to save the model both in local and remote repositories
                    use_sparse_silu=True,  # when turned on, it kills negligible swish outputs < 0.1
                    use_sparse_regularization=True,  # for sparse silu regularization
                    run_name="_sparse_pretraining",
                )

                # # "pretrained_model_name" here means a model finetuned on a downstream task that you want to build upon
                # # If you omit the parameter, it will automatically download a pretrained model (not finetuned)
                # create_model(
                #     **common_args,
                #     push_to_hub=True,  # whether to save the model both in local and remote repositories
                #     use_sparse_silu=True,  # when turned on, it kills negligible swish outputs < 0.1
                #     use_sparse_regularization=True,  # for sparse silu regularization
                #     # run_name="Mistral-7B-v0.1_cola_swiglu"
                #     run_name="_sparse_swiglu_ignore_0_1",
                #     pretrained_model_name="Mistral-7B-v0.1_cola_sparse_swiglu_scratch",
                # )

                # create_model(
                #     **common_args,
                #     push_to_hub=True,  # whether to save the model both in local and remote repositories
                #     use_sparse_silu=False,  # when turned on, it kills negligible swish outputs < 0.1
                #     use_sparse_regularization=True,  # for sparse silu regularization
                #     # run_name="Mistral-7B-v0.1_cola_swiglu"
                #     run_name="Mistral_scratch_cola",
                #     pretrained_model_name="Mistral-7B-v0.1_cola_sparse_swiglu_scratch",
                # )
