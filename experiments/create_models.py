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
import argparse

from data.get_dataset import get_dataset
from models.build_model import build_model
from models.t5_sparity_check.t5_sparsity_check import MLPRouter
from utils.constants import (
    # Models
    GPT2,
    AXB,
    GPT2_XL,
    T5,
    T5_LARGE,
    T5_3B,
    # Datasets
    # SST2,
    BILLSUM,
    SQUAD,
    QNLI,
    SST2,
    COLA,
    RTE,
    WIC,
    MULTIRC,
    BOOLQ,
)


os.environ["MASTER_PORT"] = "29509"


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)

        logits = outputs.get("logits")[:, 0]
        labels = labels.to(logits.dtype)
        # print(logits)
        # print(labels)

        # Check if there are any NaN values in logits
        if torch.isnan(logits).any():
            # If there are NaN values, return a zero loss for this batch
            # This effectively "skips" optimizing this batch.
            # The zero gradients won't affect the optimizer update.
            print("NAN")
            loss = torch.tensor(0.0).to(logits.device).requires_grad_()
        else:
            loss = F.binary_cross_entropy(logits, labels)

        # print(loss)

        return (loss, outputs) if return_outputs else loss


class MARETrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_router = True
    def compute_penalty(self, model):
        loss = None
        for is_decoder, network in enumerate(
            [model.transformer.encoder, model.transformer.decoder]
        ):
            for block_idx, block in enumerate(network.block):
                if is_decoder:
                    mlp_layer = block.layer[2].DenseReluDense
                else:
                    mlp_layer = block.layer[1].DenseReluDense
                if loss is None:
                    if mlp_layer.routing_probs is None:
                        continue
                    loss = F.mse_loss(
                        mlp_layer.routing_probs,
                        mlp_layer.activation,
                    )
                else:
                    loss += F.mse_loss(
                        mlp_layer.routing_probs,
                        mlp_layer.activation,
                    )

        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)

        logits = outputs.get("logits")[:, 0]
        labels = labels.to(logits.dtype)

        # # get penalty term
        # lambda_coeff = 1
        # regularizer = lambda_coeff * self.compute_penalty(model)

        # alpha = 0.1
        # penalty = self.compute_penalty(model)
        # loss = F.binary_cross_entropy(logits, labels)
        #
        # if penalty:
        #     loss = loss + alpha * penalty

        if self.train_router:
            loss = self.compute_penalty(model)
        else:
            loss = F.binary_cross_entropy(logits, labels)
        # loss = F.binary_cross_entropy(logits, labels)
        # loss = 1 / 20 * self.compute_penalty(model) + F.binary_cross_entropy(
        #     logits, labels
        # )  # + self.compute_penalty(model)
        # if loss is None:
        #     # loss = F.binary_cross_entropy(logits, labels)
        #     loss = F.binary_cross_entropy(torch.tensor(.0), torch.tensor(.0))
        #     print("hi loss is NOne")
        # else:
        #     print('hi loss is not NOne')
        #     # loss = loss + F.binary_cross_entropy(logits, labels)
        # print(loss)
        # print(self.compute_penalty(model))
        return (loss, outputs) if return_outputs else loss


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
    use_distillation:bool = True,
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
    dataset = get_dataset(dataset_name, tokenizer, model, model_type)
    train_dataset, val_dataset, test_dataset = dataset.get_tokenized_dataset()

    data_collator = dataset.get_data_collator()
    compute_metrics = dataset.get_compute_metrics()
    preprocess_logits_for_metrics = dataset.preprocess_logits_for_metrics

    # if "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) == 0:
    wandb_run = wandb.init(
        project=f"{model_type}_{dataset_type}",
        name=model_name,
        reinit=True,
        config=config,
    )

    training_args = TrainingArguments(
        output_dir=model_name,
        evaluation_strategy="steps",
        eval_steps=10,
        logging_steps=5,
        save_strategy="steps",
        learning_rate=5e-5,
        warmup_steps=20,
        weight_decay=0.01,
        num_train_epochs=num_epochs,
        logging_dir="exp_logs",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=model_name,
        # for memory issues
        gradient_accumulation_steps=2,
        # deepspeed="ds_config.json",
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["test_batch_size"],
        gradient_checkpointing=config["gradient_checkpointing"],
        bf16=True,
        # fp16=True,
        ddp_find_unused_parameters=True,
        hub_model_id=f"thrunlab/{model_name}",
        push_to_hub=True,
        seed=0,
        data_seed=0,
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=4, early_stopping_threshold=0.001
    )
    if use_mare_mlp:
        t = MARETrainer
    else:
        t = CustomTrainer
    # t = CustomTrainer
    trainer = t(
        # trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )

    print("\n*********************")
    print("Training ", model_name)
    print("*********************\n")

    if num_epochs != 0:
        if use_collect_stats and not use_mare_mlp:
            model.activate_stats(model)
        elif use_mare_mlp:
            # Load pre-trained weights except parameters in routers
            if True:
                print(f"thrunlab/{pre_trained_model_name}.pt")
                state_dict = torch.load(f"thrunlab/{pre_trained_model_name}.pt")
                print(state_dict.keys())
                model.load_state_dict(
                    state_dict,
                    strict=False,
                )
                # model.from_pretrained(f"thrunlab/{pre_trained_model_name}")

            # Collect Stats
            # if use_collect_stats:
            #     model.activate_stats(model, True)
            #     trainer.train_router = False
            #     print(trainer.evaluate(train_dataset, metric_key_prefix="train"))
            #     trainer.train_router = True
            model.activate_mare_mlp(model)
            # if True or not use_distillation:
            #     model.activate_mare_mlp(model)

            # Freeze model params
            for name, param in model.named_parameters():
                if "router" not in name:
                    param.requires_grad = False
                else:
                    print(name)
            # for (
            #     name,
            #     module,
            # ) in (
            #     model.transformer.named_modules()
            # ):  # @Jay: Should be removed after the experiment
            #     if "attention" not in name:
            #         for name_, param in module.named_parameters():
            #             print(f"{name_} is frozen.")
            #             param.requires_grad = False
            # for (
            #     _,
            #     module,
            # ) in (
            #     model.transformer.named_modules()
            # ):  # @Jay: Should be removed after the experiment
            #     for name, param in module.named_parameters():
            #         if "router" in name:
            #             print(f"{name} is Unfrozen.")
            #             param.requires_grad = True
        # else:
        #     model.from_pretrained(f"thrunlab/{model_name}")
        #     os.makedirs("thrunlab", exist_ok=True)
        #     torch.save(model.state_dict(), f"thrunlab/{model_name}.pt")
        #     return
        print("Training")
        trainer.train()
    if do_evaluation:
        if save_stats:
            model.activate_stats(model, True)

        if use_mare_mlp:
            model.activate_mare_mlp(model)
        if use_mare_mlp and save_stats:
            trainer.train_router = False
        print(trainer.evaluate(test_dataset, metric_key_prefix="test"))
        if save_stats:
            dir_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "mare mlp stats histograms"
            )
            filename = f"data-{dataset_name}_epochs-{num_epochs}_int ff activations_"
            model.plot_stats(model, os.path.join(dir_path, filename))
    wandb_run.finish()
    # if do_evaluation:
    #     return evaluate_model(
    #         model_name,
    #         dataset_name,
    #         model=model,
    #         tokenizer=tokenizer,
    #         test_dataset=test_dataset,
    #         trainer=trainer,
    #     )

    # if "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) == 0:
    #     wandb_run.finish()
    if push_to_hub:
        print("pushing to hub")
        trainer.push_to_hub()
        os.makedirs("thrunlab", exist_ok=True)
        torch.save(model.state_dict(), f"thrunlab/{model_name}.pt")
        # model.from_pretrained(f"thrunlab/{model_name}")

    # Empty out gpu memory
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def get_model_name(
    model_name: str,
    dataset_name: str,
    num_epochs: int = 5,
    num_generalists: int = 0,
    num_experts: int = 0,
    expert_size: int = 0,
    use_mare_mlp: bool = False,
    run_name: str = "",
):
    # Build a Gated Attention model
    model_name = f"{model_type}_{dataset_name}"
    pre_trained_model_name = model_name
    if num_generalists != 0:
        model_name += f"_mare_ar{num_generalists}_ex{num_experts}"
        if expert_size:
            model_name += f"_size-{expert_size}"
    elif num_experts:
        model_name += f"_moe_ex{num_experts}"
    else:
        model_name += f"_dense"

    model_name += f"_epochs-{num_epochs}"

    if use_mare_mlp:
        pre_trained_model_name += f"_dense_epochs-{num_epochs}"  # + "_collected-stats"
        model_name += "_mare_mlp"
    model_name += run_name  # optional
    return model_name, pre_trained_model_name


def create_model(
    model_type: str,
    dataset_name: str,
    num_epochs: int = 5,
    random_token_selection: bool = False,
    run_name: str = "",
    capacity_factor: int = 0,
    use_mare_lora: bool = False,
    use_mare_attention: bool = False,
    use_mare_upcycling: bool = False,
    use_mare_mlp: bool = False,
    use_collect_stats: bool = False,
    num_generalists: int = 0,
    num_experts: int = 0,
    expert_size: int = 0,
    apply_to_half_layers: bool = False,
    do_evaluation: bool = True,
    train_batch_size: int = 0,
    test_batch_size: int = 0,
    gradient_checkpointing: bool = False,
    save_stats: bool = False,
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
    model_name, pre_trained_model_name = get_model_name(
        model_name=model_type,
        dataset_name=dataset_name,
        num_epochs=num_epochs,
        num_generalists=num_generalists,
        num_experts=num_experts,
        expert_size=expert_size,
        use_mare_mlp=use_mare_mlp,
        run_name=run_name,
    )

    # captures the method parameters and model_name as a dictionary
    config = locals()

    model = build_model(
        model_type,
        dataset_name,
        use_mare_lora=use_mare_lora,
        use_mare_attention=use_mare_attention,
        use_mare_upcycling=use_mare_upcycling,
        use_mare_mlp=use_mare_mlp or use_collect_stats,
        num_generalists=num_generalists,
        num_experts=num_experts,
        expert_size=expert_size,
        capacity_factor=capacity_factor,
        apply_to_half_layers=apply_to_half_layers,
    )

    model.config.pad_token_id = model.config.eos_token_id

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    if GPT2 in model_type:
        tokenizer.pad_token = tokenizer.eos_token

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
        pre_trained_model_name=pre_trained_model_name,
    )


def download_model(
    model_name: str, dataset_name: str, num_epochs: int, force_download: bool = False
):
    model = build_model(
        model_type,
        dataset_name,
    )
    model_name, _ = get_model_name(
        model_name=model_name, dataset_name=dataset_name, num_epochs=num_epochs,
    )
    if not os.path.exists(f"thrunlab/{model_name}.pt"):
        model.from_pretrained(f"thrunlab/{model_name}")
        os.makedirs("thrunlab", exist_ok=True)
        torch.save(model.state_dict(), f"thrunlab/{model_name}.pt")


if __name__ == "__main__":
    torch.set_printoptions(threshold=300)
    # parser = argparse.ArgumentParser(description="")
    # parser.add_argument("--datasets", nargs='+', type=str, help="A list of datasets")
    #
    # args = parser.parse_args()


    dataset_types = [
        # BILLSUM,
        COLA,
        # AXB,
        # BOOLQ,
        # MULTIRC,
        # WIC,
    ]
    # if hasattr(args, "datasets"):
    #     dataset_types = args.datasets

    total_sizes = {
        T5: 3072,
        T5_LARGE: 4096,
        T5_3B: 16384,
    }

    model_types = [T5_LARGE]
    num_epochs = 5
    for model_type in model_types:
        for dataset_type in dataset_types:
            common_args = {
                "model_type": model_type,
                "dataset_name": dataset_type,
                "num_epochs": num_epochs,
                "train_batch_size": 32,
                "test_batch_size": 64,
                "gradient_checkpointing": False,
            }
            # Download model in local folder
            # download_model(
            #     model_name=model_type,
            #     dataset_name=dataset_type,
            #     num_epochs=num_epochs,
            # )
            print(f"{get_model_name(model_type, dataset_type, num_epochs)[0]}.pt")
            if False: #not os.path.exists(f"{get_model_name(model_type, dataset_type, num_epochs)[0]}.pt"): # not os.path.exists(f"{get_model_name(model_type, dataset_type, num_epochs)}.pt"):
                create_model(
                    # run_name="dense_epochs5",
                    **common_args,
                    push_to_hub=True,
                )
            # create_model(
            #     use_mare_mlp=False,
            #     **common_args,
            #     save_stats=True,
            #     use_collect_stats=True,
            #     push_to_hub=True,
            #     num_generalists=16,
            #     num_experts=48,
            #     expert_size=32,
            # )
            total_size = total_sizes[model_type]
            expert_size = 16
            for fixed_cost in [
                0.2, 0.3,
            ]:
                for ratio in [0,  0.5, 0.9, 1.0]:
                    total_nums = int(total_size / expert_size * fixed_cost)
                    total_nums += total_nums % 2
                    num_generalists = int(total_nums * ratio) + int(
                        int(total_nums * ratio) % 2
                    )  # To make it even
                    num_experts = total_nums - num_generalists
                    create_model(
                        use_mare_mlp=True,
                        **common_args,
                        save_stats=False,
                        use_collect_stats=True,
                        push_to_hub=False,
                        num_generalists=num_generalists,
                        num_experts=num_experts,
                        expert_size=expert_size,
                    )

            # create_model(
            #     use_mare_upcycling=True,
            #     capacity_factor=2,
            #     num_generalists=1,
            #     num_experts=2,
            #     apply_to_half_layers=True,
            #     **common_args,
            # )

            # create_model(
            #     use_mare_upcycling=True,
            #     capacity_factor=2,
            #     num_generalists=0,
            #     num_experts=4,
            #     apply_to_half_layers=True,
            #     **common_args,
            # )
            #

            # create_model(
            #     use_mare_upcycling=True,
            #     capacity_factor=2,
            #     num_generalists=1,
            #     num_experts=15,
            #     apply_to_half_layers=True,
            #     **common_args,
            # )

            # create_model(
            #     use_mare_upcycling=True,
            #     capacity_factor=2,
            #     num_generalists=0,
            #     num_experts=16,
            #     apply_to_half_layers=True,
            #     **common_args,
            # )

    # Original dense
    # create_model(
    #     run_name="baseline",
    #     **common_args,
    # )

    # MARE without experts
    # create_model(
    #     use_mare_upcycling=True,
    #     num_generalists=2,
    #     capacity_factor=2,
    #     num_experts=0,
    #     run_name="c2_dense_2_half",
    #     apply_to_half_layers=True,
    #     **common_args,
    # )

    # # MARE without ARs
    # create_model(
    #     use_mare_upcycling=True,
    #     capacity_factor=2,
    #     num_generalists=0,
    #     num_experts=8,
    #     run_name="moe_ex8_one_more",
    #     apply_to_half_layers=True,
    #     **common_args,
    # )

    # MARE
    # create_model(
    #     use_mare_upcycling=True,
    #     capacity_factor=2,
    #     num_generalists=1,
    #     num_experts=7,
    #     run_name="mare_ar1_ex7_half_from_ft_scaler_per_expert",
    #     apply_to_half_layers=True,
    #     **common_args,
    # )

    # create_model(
    #     use_mare_upcycling=True,
    #     capacity_factor=2,
    #     num_generalists=1,
    #     num_experts=4,
    #     run_name="c2_mare_ar1_ex4_half_from_ft_dense",
    #     apply_to_half_layers=True,
    #     **common_args,
    # )
