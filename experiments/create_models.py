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
    # Accelerators
    SVD,
    # Models
    GPT2,
    GPT2_MEDIUM,
    GPT2_XL,
    # Parameters
    NUM_BLOCKS_GPT2,
    NUM_BLOCKS_GPT2_MEDIUM,
    # Datasets
    # BILLSUM,
    # SST2,
    SAMSUM,
    BILLSUM,
)


def finetune(
    model_name: str,
    model,
    tokenizer,
    dataset_name: str,
    num_epochs: int = 10,
    resume_from_checkpoint: bool = False,
    random_token_selection: bool = False,
    run_name: str = "",
    learning_rate: float = 0.001,
    do_evaluation: bool = True,
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
    dataset = get_dataset(dataset_name, tokenizer, model)
    train_dataset, test_dataset = dataset.get_tokenized_dataset()
    data_collator = dataset.get_data_collator()
    # compute_metrics = dataset.get_compute_metrics()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "checkpoints", model_name)

    wandb.init(project="samsum_mare_vs_full_expert", name=run_name)

    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=25,
        save_strategy="steps",
        learning_rate=learning_rate,
        warmup_steps=500,
        per_device_train_batch_size=24,  # 32 for SMALL, 12 for MED
        per_device_eval_batch_size=64,  # 32 for MED, XL, 64 for SMALL
        weight_decay=0.01,
        num_train_epochs=num_epochs,
        logging_dir="exp_logs",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=run_name,
        # for memory issues
        # gradient_accumulation_steps=2,
        # gradient_checkpointing=True,
        fp16=True,
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
        # compute_metrics=compute_metrics,
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


def create_model(
    model_type: str,
    dataset_name: str,
    learning_rate: float,
    num_epochs: int = 5,
    random_token_selection: bool = False,
    run_name: str = "",
    use_mare_lora: bool = False,
    use_mare_attention: bool = False,
    use_lora: bool = False,
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
    model_name = model_type
    if use_lora:
        model_name += "_lora"
    if use_mare_attention:
        model_name += "_gated"

    model = build_model(
        model_type,
        use_mare_lora=use_mare_lora,
        use_lora=use_lora,
        use_mare_attention=use_mare_attention,
        random_token_selection=random_token_selection,
        **kwargs,
    )
    model.config.pad_token_id = model.config.eos_token_id

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.pad_token = tokenizer.eos_token

    finetune(
        model_name,
        model,
        tokenizer,
        dataset_name,
        num_epochs,
        resume_from_checkpoint=False,
        learning_rate=learning_rate,
        run_name=run_name,
        random_token_selection=random_token_selection,
    )


if __name__ == "__main__":
    common_args = {
        "model_type": GPT2_MEDIUM,
        "dataset_name": SAMSUM,
        "num_epochs": 6,
        "learning_rate": 5e-5,
    }

    original = {
        "run_name": "baseline",
    }

    mare = {
        "use_mare_attention": True,
        "num_generalists": 2,
        "run_name": "ar2_1/7",
    }

    moe = {
        "use_mare_attention": True,
        "num_generalists": 0,
        "run_name": "ar0_1/4",
    }

    moelora = {
        "use_mare_lora": True,
        "run_name": "marelora_rank4_ar0_ex8_1/4",
        "num_generalists": 0,
        "num_experts": 8,
    }

    marelora = {
        "use_mare_lora": True,
        "run_name": "marelora_rank4_ar1_ex_4_1/4",
        "num_generalists": 1,
        "num_experts": 4,
    }

    exps_to_run = [
        # original,
        # mare,
        moelora,
        marelora,
    ]

    for exp in exps_to_run:
        create_model(**common_args, **exp)
