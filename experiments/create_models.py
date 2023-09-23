import os
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoTokenizer,
)
import wandb

from data.get_dataset import get_dataset
from evaluate_models import evaluate_model
from models.build_model import build_model
from utils.constants import (
    # Models
    GPT2,
    T5,
    # Datasets
    # SST2,
    SAMSUM,
    BILLSUM,
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
    config=None,
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
    :returns: Dictionary of evaluation metrics and inference time if `do_evaluation` is true
    """
    print("Finetuning started...")
    dataset = get_dataset(dataset_name, tokenizer, model_type)
    train_dataset, test_dataset = dataset.get_tokenized_dataset()
    data_collator = dataset.get_data_collator()
    compute_metrics = dataset.get_compute_metrics()
    preprocess_logits_for_metrics = dataset.preprocess_logits_for_metrics

    init_wandb = True
    try:
        init_wandb = int(os.environ["LOCAL_RANK"]) == 0
    except:
        pass

    if init_wandb:
        wandb_run = wandb.init(
            project="t5_billsum",
            name=run_name,
            reinit=True,
            config=config,
        )

    training_args = TrainingArguments(
        output_dir=model_name,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        save_strategy="steps",
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        num_train_epochs=num_epochs,
        logging_dir="exp_logs",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=run_name,
        # for memory issues
        gradient_accumulation_steps=2,
        deepspeed="ds_config.json",
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["test_batch_size"],
        gradient_checkpointing=config["gradient_checkpointing"],
        bf16=True,
        push_to_hub=True,
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=10, early_stopping_threshold=0.001
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[early_stopping],
    )

    print("\n*********************")
    print("Training ", model_name)
    print("*********************\n")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if do_evaluation:
        return evaluate_model(
            model_name,
            dataset_name,
            model=model,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            trainer=trainer,
        )

    if int(os.environ["LOCAL_RANK"]) == 0:
        wandb_run.finish()


def create_model(
    model_type: str,
    dataset_name: str,
    num_epochs: int = 5,
    # random_token_selection: bool = False,
    run_name: str = "",
    capacity_factor: int = 0,
    use_mare_lora: bool = False,
    use_mare_attention: bool = False,
    apply_to_half_layers: bool = False,
    do_evaluation: bool = True,
    **kwargs,
):
    """
    Load a naive pretrained model from HuggingFace, apply the accelerator to FC layers,
    and finetune it on the specified dataset.

    :param model_type: Type of the model (TXL, GPT, etc.)
    :param dataset_name: Name of the dataset to train on
    :param num_epochs: Number of epochs to finetune the model
    :param do_evaluation: Whether to evaluate the model after training and save the metrics
    :return: Dictionary of evaluation logs if `do_evaluation` is true
    """
    # Build a Gated Attention model
    model_name = model_type + "_" + run_name
    # captures the method parameters and model_name as a dictionary
    config = locals()

    model = build_model(
        model_type,
        dataset_name,
        use_mare_lora=use_mare_lora,
        use_mare_attention=use_mare_attention,
        capacity_factor=capacity_factor,
        apply_to_half_layers=apply_to_half_layers,
        **kwargs,
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
    )


if __name__ == "__main__":
    common_args = {
        "model_type": T5,
        "dataset_name": BILLSUM,
        "num_epochs": 6,
        "train_batch_size": 16,
        "test_batch_size": 32,
        "gradient_checkpointing": True,
    }

    # Original dense
    # create_model(
    #     run_name="baseline",
    #     **common_args,
    # )

    # MARE without experts
    create_model(
        use_mare_upcycling=True,
        num_generalists=2,
        capacity_factor=2,
        num_experts=0,
        run_name="c2_dense_2_half",
        apply_to_half_layers=True,
        **common_args,
    )

    # MARE
    create_model(
        use_mare_upcycling=True,
        capacity_factor=2,
        num_generalists=1,
        num_experts=16,
        run_name="c2_mare_ar1_ex8_half",
        apply_to_half_layers=True,
        **common_args,
    )

    # MARE without ARs
    create_model(
        use_mare_upcycling=True,
        num_generalists=2,
        num_experts=0,
        run_name="moe_ex8",
        **common_args,
    )
