from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding
import evaluate
import torch
from datasets import load_dataset
import numpy as np
import pickle
import os

from .dataset import Dataset
from utils.constants import (
    QNLI,
    SST2,
    COLA,
    RTE,
    MRPC,
    WIC,
    MULTIRC,
    BOOLQ,
    SUPERGLUE,
)


class GLUEDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        model,
        model_type: str,
        dataset_type: str,
    ):
        """
        :param tokenizer: Tokenizer to tokenize the inputs
        :param model: Model to finetune / evaluate
        :param metrics: Metrics to evaluate the model
        :param prefix: Prefix to the inputs to start the summarization task
        """
        super().__init__(tokenizer, model, model_type)
        self.model_type = model_type.split("/")[-1]
        self.dataset_type = dataset_type
        self.metrics = evaluate.load("accuracy")
        self.benchmark = "super_glue" if dataset_type in SUPERGLUE else "glue"

    def get_tokenized_dataset(self):
        """
        Load and build a simple summarization dataset
        :return: tokenized dataset, data collator, and compute_metrics function
        """
        train_path = f"datasets/{self.model_type}_{self.dataset_type}_train_dataset_classification_16.pkl"
        test_path = f"datasets/{self.model_type}_{self.dataset_type}_test_dataset_classification_16.pkl"
        validation_path = f"datasets/{self.model_type}_{self.dataset_type}_validation_dataset_classification_16.pkl"

        try:
            # Try to load preprocessed data
            with open(train_path, "rb") as f:
                train_dataset = pickle.load(f)
            with open(test_path, "rb") as f:
                test_dataset = pickle.load(f)
            with open(validation_path, "rb") as f:
                val_dataset = pickle.load(f)
            print("Loaded preprocessed data from disk.")

        except FileNotFoundError:
            # If preprocessed data does not exist, load and preprocess
            print("No dataset found.")
            dataset = load_dataset(
                self.benchmark,
                self.dataset_type,
                # streaming=True
            )

            max_data_points = 16
            dataset = dataset.select(range(3 * max_data_points))
            tokenized_dataset = dataset.map(
                self.preprocess,
                remove_columns=dataset.column_names["train"],
            )

            print("Tokenization completed.")

            train_dataset = tokenized_dataset.select(range(max_data_points))
            test_dataset = tokenized_dataset.select(
                range(max_data_points, 2 * max_data_points)
            )
            val_dataset = tokenized_dataset.select(
                range(2 * max_data_points, 3 * max_data_points)
            )

            # # GLUE doesn't provide answers for TEST, so we have to manually make them
            # train_test_dataset = tokenized_dataset["train"].train_test_split(
            #     test_size=0.1
            # )

            # train_dataset = train_test_dataset["train"]
            # test_dataset = train_test_dataset["test"]
            # val_dataset = tokenized_dataset["validation"]

            # Save the preprocessed data
            with open(train_path, "wb") as f:
                pickle.dump(train_dataset, f)
            with open(test_path, "wb") as f:
                pickle.dump(test_dataset, f)
            with open(validation_path, "wb") as f:
                pickle.dump(val_dataset, f)
            print("Preprocessed and saved data to disk.")

        return train_dataset, val_dataset, test_dataset

    def get_data_collator(self):
        """
        Load the data collator that is responsible for taking in batches of examples
        and converting them into a format that can be consumed by the model.
        """
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            max_length=512,
        )
        return data_collator

    def compute_metrics(self, preds):
        """
        Load the compute_metrics function that is used by the HuggingFace Trainer to
        evaluate the model.
        :return:
        """
        preds, labels = preds
        preds = preds.argmax(-1)

        # preds = np.array(
        #     preds[0][:, 0] > 1 / 2, dtype=float
        # )  # the output is sigmoid
        # preds = np.array(preds > 1 / 2, dtype=float)  # the output is sigmoid
        return self.metrics.compute(predictions=preds, references=labels)

    def preprocess(self, example):
        if self.dataset_type == QNLI:
            input_text = (
                "Does the context entail the question? \n"
                + "context: "
                + example["sentence"]
                + " question: "
                + example["question"]
            )
        elif self.dataset_type == SST2:
            input_text = (
                "Positive or negative sentiment?\n" + example["sentence"]
            )
        elif self.dataset_type == RTE:
            input_text = (
                "Entailed?\n"
                + "\First sentence: "
                + example["sentence1"]
                + "\Second sentence: "
                + example["sentence2"]
            )
        elif self.dataset_type == MRPC:
            input_text = (
                "Equivalent?\n"
                + "\First sentence: "
                + example["sentence1"]
                + "\Second sentence: "
                + example["sentence2"]
            )
        elif self.dataset_type == COLA:
            input_text = "cola classification: " + example["sentence"]
        elif self.dataset_type == BOOLQ:
            input_text = (
                "Yes or No?"
                + "\nQuestion: "
                + example["question"]
                + "\nPessage: "
                + example["passage"]
            )
        elif self.dataset_type == MULTIRC:
            input_text = (
                "Is the answer correct?"
                + "\nPassage: "
                + example["paragraph"]
                + "\nQuestion: "
                + example["question"]
                + "\nAnswer: "
                + example["answer"]
            )
        elif self.dataset_type == WIC:
            input_text = (
                "Is the word used in the same context?"
                + "\nFirst sentence: "
                + example["sentence1"]
                + "\nSecond sentence: "
                + example["sentence2"]
            )
        else:
            raise ValueError(f"Invalid dataset: ({self.dataset_type})")

        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        # inputs["labels"] = "YES" if example["label"] == 1 else "NO"
        inputs["labels"] = example["label"]

        return inputs

    def preprocess_logits_for_metrics(self, logits, labels):
        """
        Original Trainer may have a memory leak.
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        # pred_ids = torch.argmax(logits[0], dim=-1)
        # print(logits)
        pred_ids = torch.argmax(logits, dim=-1)
        # if "mistral" in self.model_type:
        #     pred_ids = logits[:, 0]
        # else:
        #     pred_ids = logits[0][:, 0]
        return pred_ids
