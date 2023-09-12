import os
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    GPT2ForSequenceClassification,
    AutoTokenizer,
    TrainerCallback,
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
    learning_rate: float = 0.001,
    do_evaluation: bool = True,
    run_name: str = None,
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
    compute_metrics = dataset.get_compute_metrics()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "checkpoints", model_name)

    class PrintGradientsCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            model = kwargs["model"]
            for name, param in model.named_parameters():
                # if "router" in name:
                print(f"Gradient of {name} is {param.grad}")

    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=25,
        save_strategy="steps",
        learning_rate=learning_rate,
        warmup_steps=500,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=64,  # 32 for MED, XL, 64 for SMALL
        weight_decay=0.005,
        num_train_epochs=num_epochs,
        logging_dir="exp_logs",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=run_name,
        # for memory issues
        gradient_accumulation_steps=2,
        # gradient_checkpointing=True,
        fp16=True,
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=5, early_stopping_threshold=0.001
    )

    gradient_logger = PrintGradientsCallback()

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
    use_gated_attention: bool,
    use_lora: bool,
    model_type: str,
    dataset_name: str,
    learning_rate: float,
    num_epochs: int = 5,
    run_name: str = None,
    random_router: bool = False,
    do_evaluation: bool = True,
    freeze_layers: bool = False,
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
    if use_gated_attention:
        model_name += "_gated"
    run_name = model_name + "_" + dataset_name + "_" + run_name

    wandb.init(
        project="gated_attention_billsum",
        name=run_name,
        config={
            "K": "T / (self.layer_idx - self.config.n_layer // 12 + 3)), 6)",
            "random_router": random_router,
            "model_name": model_name,
            "dataset_name": dataset_name,
        },
    )

    model = build_model(
        model_type,
        use_lora,
        use_gated_attention,
        random_router,
        freeze_layers=freeze_layers,
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
        run_name=run_name,
        resume_from_checkpoint=False,
        learning_rate=learning_rate,
    )


if __name__ == "__main__":
    original = {
        "use_lora": False,
        "use_gated_attention": False,
        "model_type": GPT2_MEDIUM,
        "dataset_name": BILLSUM,
        "num_epochs": 6,
        "learning_rate": 1e-4,
        "run_name": "original",
    }

    gated = {
        "use_lora": False,
        "use_gated_attention": True,
        "model_type": GPT2_MEDIUM,
        "dataset_name": BILLSUM,
        "num_epochs": 7,
        "learning_rate": 1e-4,
        "run_name": "_noise",
        "freeze_layers": False,
    }

    gated_random = {
        "use_lora": False,
        "use_gated_attention": True,
        "model_type": GPT2_MEDIUM,
        "dataset_name": BILLSUM,
        "num_epochs": 6,
        "learning_rate": 1e-4,
        "random_router": True,
        "run_name": "random_higher_lr",
    }

    exps_to_run = [
        # original,
        gated,
        # gated_random,
    ]

    for exp in exps_to_run:
        create_model(**exp)
