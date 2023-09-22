from .get_billsum_dataset import Billsum
from .get_squad_dataset import SQuAD
from .get_openwebtext_dataset import OpenWebText
from .get_sst2_dataset import SST2_Dataset
from .get_samsum_dataset_ import Samsum
from utils.constants import BILLSUM, SQUAD, OPENWEBTEXT, SST2, SAMSUM, QNLI, COLA

from transformers import (
    T5TokenizerFast,
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset

import torch


def convert_to_t5_format(example, task, tokenizer):
    if task == "qnli":
        source_text = (
            "question: " + example["question"] + " context: " + example["sentence"]
        )
        target_text = "entailment" if example["label"] == 1 else "not_entailment"
    elif task == "sst2":
        source_text = example["sentence"]
        target_text = "positive" if example["label"] == 1 else "negative"
    elif task == "cola":
        source_text = example["sentence"]
        target_text = "acceptable" if example["label"] == 1 else "unacceptable"
    elif task == "squad":
        source_text = (
            "question: " + example["question"] + " context: " + example["context"]
        )
        target_text = example["answers"]["text"]
    else:
        raise ValueError("Invalid task")
    return {
        "input_text": source_text,
        "target_text": target_text,
        "input_ids": tokenizer(
            source_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        ).input_ids[0],
        "labels": tokenizer(
            target_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        ).input_ids[0],
    }


def get_dataset(dataset_name: str, model_name: str, tokenizer, model):
    """
    Tokenize a dataset and get additional data collator and metrics necessary for training the models

    :param dataset_name: Name of the dataset to build
    :param tokenizer: Tokenizer to tokenize the inputs
    :param model: Model to finetune / evaluate
    :return: tokenized dataset, data collator, and compute_metrics function
    """
    if "t5" in model_name:
        if dataset_name in [QNLI, SST2, COLA]:
            dataset = load_dataset("glue", dataset_name)
        else:
            dataset = load_dataset(dataset_name)
        dataset = dataset.map(
            lambda x: convert_to_t5_format(x, dataset, tokenizer),
            remove_columns=dataset.column_names["train"],
        )

    else:
        if dataset_name == BILLSUM:
            dataset = Billsum(tokenizer, model)
        elif dataset_name == SQUAD:
            dataset = SQuAD(tokenizer, model)
        elif dataset_name == OPENWEBTEXT:
            dataset = OpenWebText(tokenizer, model)
        elif dataset_name == SST2:
            dataset = SST2_Dataset(tokenizer, model)
        elif dataset_name == SAMSUM:
            dataset = Samsum(tokenizer, model)
        else:
            assert False, f"No dataset named {dataset_name} available."

    return dataset
