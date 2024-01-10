import os
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoTokenizer,
)
from torch.nn import functional as F
import wandb
import copy

import gc
import torch

from data.get_dataset import get_dataset
from models.build_model import build_model
from experiments.models.sparse_mistral.relufication import (
    ReluficationRateScheduler,
)
from utils.constants import (
    # Models
    GPT2,
    GPT2_XL,
    T5,
    T5_LARGE,
    T5_3B,
    MISTRAL_7B,
    # Datasets
    # SST2,
    BILLSUM,
    SQUAD,
    QNLI,
    SST2,
    COLA,
    RTE,
    MRPC,
    WIC,
    MULTIRC,
    BOOLQ,
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
    pre_trained_model_name: str = None,
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
    train_dataset = train_dataset.select(range(128))
    val_dataset = train_dataset
    test_datset = train_dataset

    data_collator = dataset.get_data_collator()
    compute_metrics = dataset.get_compute_metrics()

    if True or (
        "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) == 0
    ):
        wandb_run = wandb.init(
            project=f"{model_name}_{dataset_type}_mlp",
            name=model_name,
            reinit=True,
            config=config,
        )

    training_args = TrainingArguments(
        output_dir=os.path.join("ckpt", model_name),
        evaluation_strategy="steps",
        eval_steps=10,
        logging_steps=1,
        save_strategy="steps",
        learning_rate=1e-5,
        max_steps=750,
        weight_decay=0.01,
        num_train_epochs=num_epochs,
        logging_dir="exp_logs",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=model_name,
        # for memory issues
        gradient_accumulation_steps=2,
        deepspeed="ds_config.json",
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["test_batch_size"],
        gradient_checkpointing=config["gradient_checkpointing"],
        bf16=True,
        # fp16=True,
        ddp_find_unused_parameters=True,
        hub_model_id=f"thrunlab/{model_name}",
        push_to_hub=True,
        seed=2,
        data_seed=1,
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
    relufication_scheduler = ReluficationRateScheduler()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping, relufication_scheduler],
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )

    print("\n*********************")
    print("Training ", model_name)
    print("*********************\n")

    if num_epochs != 0:
        if use_collect_stats and not use_mare_mlp:
            model.activate_stats(model)
        elif use_mare_mlp:
            # model.from_pretrained(f"thrunlab/{pre_trained_model_name}")
            print(f"thrunlab/{pre_trained_model_name}")
            # state_dict = torch.load(f"thrunlab/{pre_trained_model_name}.pt")

            # model.load_state_dict(
            #     state_dict,
            #     strict=False,
            # )

            # Collect Stats
            if use_collect_stats:
                model.activate_stats(model, True)
                print("Evaluating")
                trainer.evaluate(train_dataset)
            model.activate_mare_mlp(model)

    print("Training")
    trainer.train()

    if do_evaluation:
        if save_stats:
            model.activate_stats(model, True)
        trainer.evaluate()
        if save_stats:
            dir_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "mare mlp stats histograms",
            )
            filename = (
                f"data-{dataset_name}_epochs-{num_epochs}_int ff activations_"
            )
            model.plot_stats(model, os.path.join(dir_path, filename))
            print("saved at: ", os.path.join(dir_path, filename))

    results = trainer.evaluate(test_dataset)
    print("EVALUATION: ", results)
    wandb_run.finish()

    if True or (
        "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) == 0
    ):
        wandb_run.finish()
        if push_to_hub:
            print("pushing to hub")
            trainer.push_to_hub()
            os.makedirs("thrunlab", exist_ok=True)
            torch.save(model.state_dict(), f"thrunlab/{model_name}.pt")
        # model.from_pretrained(f"thrunlab/{model_name}")

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
    save_stats: bool = False,
    use_relu: bool = False,
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

    if use_mare_mlp:
        pre_trained_model_name += (
            f"_dense_epochs-{num_epochs}"  # -{num_epochs}
        )
        model_name += "_router_from_pretrained_alpha1"
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
        use_relu=use_relu,
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
        # pre_trained_model_name=pre_trained_model_name,
    )


if __name__ == "__main__":
    dataset_types = [
        # BILLSUM,
        # SST2,
        COLA,
        # RTE,
        # MRPC,
        # QNLI,
        # BOOLQ,
        # MULTIRC,
        # WIC,
    ]

    total_sizes = {
        T5: 3072,
        T5_LARGE: 4096,
        T5_3B: 16384,
    }

    model_types = [MISTRAL_7B]
    for i in range(1):
        print("EXPERIMENT: ", i)
        for model_type in model_types:
            for dataset_type in dataset_types:
                common_args = {
                    "model_type": model_type,
                    "dataset_name": dataset_type,
                    "num_epochs": 10,
                    "train_batch_size": 64,
                    "test_batch_size": 64,
                    "gradient_checkpointing": False,
                }

                create_model(
                    **common_args,
                    push_to_hub=True,
                    use_relu=True,
                    run_name="_relu",
                )

                # total_size = total_sizes[model_type]
                # expert_size = 16
                # fixed_costs = [
                #     0.05,
                #     0.1,
                #     # 0.3,
                # ]
                # generalist_ratios = [0.0]

                # print("______MARE_______")

                # for fixed_cost in fixed_costs:
                #     for generalist_ratio in generalist_ratios:
                #         # continue
                #         total_nums = int(total_size / expert_size * fixed_cost)
                #         num_generalists = (
                #             int(total_nums * generalist_ratio) * 2 // 2
                #         )  # To make it even
                #         num_experts = total_nums - num_generalists
                #         create_model(
                #             use_mare_mlp=True,
                #             **common_args,
                #             save_stats=False,
                #             use_collect_stats=False,
                #             push_to_hub=False,
                #             num_generalists=num_generalists,
                #             num_experts=num_experts,
                #             expert_size=expert_size,
                #             fixed_costs=fixed_cost,
                #             generalist_ratios=generalist_ratio,
                #         )
