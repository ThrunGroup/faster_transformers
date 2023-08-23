import os
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from typing import List

from accelerators.apply_accelerator import apply_accelerator
from data.get_dataset import get_dataset
from evaluate_models import evaluate_model
from utils.parse_string import parse_string, dict_to_string
from models.gpt2_from_qkv import build_gated_attention_model
from utils.constants import (
    # Accelerators
    SVD,
    # Models
    GPT2,
    GPT2_MEDIUM,
    # Parameters
    NUM_BLOCKS_GPT2,
    NUM_BLOCKS_GPT2_MEDIUM,
    # Datasets
    BILLSUM,
)


def finetune(
        model_name: str,
        model,
        tokenizer,
        dataset_name: str,
        num_epochs: int = 10,
        resume_from_checkpoint: bool = False,
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
    compute_metrics = dataset.get_compute_metrics()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "checkpoints", model_name)

    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        num_train_epochs=num_epochs,
        logging_dir="exp_logs",
        save_total_limit=1,
        load_best_model_at_end=True,
        # load the best model when finished training
        metric_for_best_model="rouge1",
        # use rouge1 as the metric to determine best model
        greater_is_better=True,
        # the higher the rouge score, the better the model
        gradient_accumulation_steps=4,
        fp16=True,
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=8, early_stopping_threshold=0.0001
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
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
        num_epochs: int = 5,
        do_evaluation: bool = True,
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
    # Load a custom model with GPT2
    model = build_gated_attention_model()


if __name__ == "__main__":
    # Train a model without any accelerator
    # create_model(model_type=GPT2,
    #              dataset_name=BILLSUM,
    #              num_epochs=30,
    #              layers_to_freeze=f"0-{NUM_BLOCKS_GPT2 - 2}")
    #
    # # # Only train the accelerated FC layer and the very final linear layer
    # create_model(model_type=GPT2,
    #              dataset_name=BILLSUM,
    #              num_epochs=30,
    #              layers_to_freeze=f"0-{NUM_BLOCKS_GPT2 - 3}",
    #              layers_to_accelerate=f"{NUM_BLOCKS_GPT2 - 3}-{NUM_BLOCKS_GPT2 - 1}",
    #              train_accelerated_layers=True,
    #              accelerator_type=SVD,
    #              k=64)

    create_model(
        model_type=GPT2,
        dataset_name=BILLSUM,
        num_epochs=10,
        layers_to_freeze=f"0-{NUM_BLOCKS_GPT2 - 4}",
    )

    create_model(
        model_type=GPT2_MEDIUM,
        dataset_name=BILLSUM,
        num_epochs=30,
        layers_to_freeze=f"0-{NUM_BLOCKS_GPT2_MEDIUM - 4}",  # GPT2_MEDIUM
        layers_to_accelerate=f"{NUM_BLOCKS_GPT2_MEDIUM - 3}-{NUM_BLOCKS_GPT2_MEDIUM - 1}",
        # GPT2_MEDIUM
        train_accelerated_layers=True,
        accelerator_type=SVD,
        k=64,
    )
