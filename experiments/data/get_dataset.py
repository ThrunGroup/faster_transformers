from .dataset import Dataset
from .get_billsum_dataset import Billsum
from .get_squad_dataset import SQuAD
from .get_openwebtext_dataset import OpenWebText
from .get_sst2_dataset import SST2_Dataset
from .get_samsum_dataset_ import Samsum
from .get_default_dataset import DefaultDataset
from utils.constants import (
    BILLSUM,
    SQUAD,
    OPENWEBTEXT,
    SST2,
    SAMSUM,
    QNLI,
    COLA,
)

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
        labels = ["entailment", "not_entailment"]
    elif task == "sst2":
        source_text = example["sentence"]
        target_text = "positive" if example["label"] == 1 else "negative"
        labels = ["positive", "negative"]
    elif task == "cola":
        source_text = example["sentence"]
        target_text = "acceptable" if example["label"] == 1 else "unacceptable"
        labels = ["acceptable", "unacceptable"]
    elif task == "squad":
        source_text = (
            "question: " + example["question"] + " context: " + example["context"]
        )
        target_text = example["answers"]["text"]
        labels = []
    else:
        raise ValueError(f"Invalid task ({task})")

    special_tokens_dict = {"additional_special_tokens": labels}
    tokenizer.add_special_tokens(special_tokens_dict)

    target_tokenized = tokenizer(
        target_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    labels = target_tokenized.input_ids[0]
    labels[labels == tokenizer.pad_token_id] = -100
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
        "labels": labels,
        "labels_attention_mask": target_tokenized.attention_mask[0],
        "decoder_attention_mask": target_tokenized.attention_mask[0],
    }


def get_dataset(dataset_name: str, model_type: str, tokenizer, model):
    """
    Tokenize a dataset and get additional data collator and metrics necessary for training the models

    :param dataset_name: Name of the dataset to build
    :param model_name: Name of the model
    :param tokenizer: Tokenizer to tokenize the inputs
    :param model: Model to finetune / evaluate
    :return: tokenized dataset, data collator, and compute_metrics function
    """
    if "t5" in model_type and dataset_name in [QNLI, SST2, COLA, SQUAD]:
        if dataset_name in [QNLI, SST2, COLA]:
            dataset = load_dataset("glue", dataset_name)
        else:
            dataset = load_dataset(dataset_name)
        tokenized_dataset = dataset.map(
            lambda x: convert_to_t5_format(x, dataset_name, tokenizer),
            remove_columns=dataset.column_names["train"],
        )
        model.resize_token_embeddings(
            len(tokenizer)
        )  # Resize model vocab after adding labels(e.g. entailment) to vocab
        dataset = DefaultDataset(tokenizer, model, model_type)

        def get_tokenized_dataset(self):
            return tokenized_dataset["train"], tokenized_dataset["validation"]

        def preprocess(self):
            return None

        def compute_metrics(self):
            return None

        def get_data_collator(self):
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                padding=True,  # Adaptive to batch's longest sequence length
                # padding = "max_length",
                max_length=512,
            )
            return data_collator

        # Override functions defined in class
        dataset.get_tokenized_dataset = get_tokenized_dataset.__get__(dataset, Dataset)
        dataset.preprocess = preprocess.__get__(dataset, Dataset)
        dataset.compute_metrics = compute_metrics.__get__(dataset, Dataset)
        dataset.get_data_collator = get_data_collator.__get__(dataset, Dataset)

    else:
        if dataset_name == BILLSUM:
            dataset = Billsum(tokenizer, model, model_type)
        elif dataset_name == SQUAD:
            dataset = SQuAD(tokenizer, model, model_type)
        elif dataset_name == OPENWEBTEXT:
            dataset = OpenWebText(tokenizer, model, model_type)
        elif dataset_name == SST2:
            dataset = SST2_Dataset(tokenizer, model, model_type)
        elif dataset_name == SAMSUM:
            dataset = Samsum(tokenizer, model, model_type)
        else:
            assert False, f"No dataset named {dataset_name} available."

    return dataset
