from transformers import DataCollatorForLanguageModeling, AutoTokenizer, \
    GPT2Model, GPT2Config
import evaluate
import numpy as np
from datasets import load_dataset
from itertools import chain

from .dataset import Dataset


class Concatenator(object):
    def __init__(self, chunk_size=2048):
        self.chunk_size = chunk_size
        self.residual = {"input_ids": [], "attention_mask": []}

    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k])) for k, v in self.residual.items()
        }

        total_length = len(
            concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i: i + self.chunk_size]
                    for i in
                    range(0, chunk_num * self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size):]
                for k, v in concatenated_samples.items()
            }
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys()}

        result["labels"] = result["input_ids"].copy()

        return result


class Samsum(Dataset):
    def __init__(self, tokenizer, model, metrics: str = "rouge"):
        """
        :param tokenizer: Tokenizer to tokenize the inputs
        :param model: Model to finetune / evaluate
        :param metrics: Metrics to evaluate the model
        :param prefix: Prefix to the inputs to start the summarization task
        """
        super().__init__(tokenizer, model)
        self.tokenizer = tokenizer
        self.metrics = evaluate.load(metrics)  # Load the metrics to be used by compute_metrics
        self.prompt = (
            f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
            f"{{summary}}{{eos_token}}"
        )


    def get_tokenized_dataset(self):
        """
        Load and build a simple summarization dataset
        :return: tokenized dataset, data collator, and compute_metrics function
        """
        dataset = load_dataset("samsum", split="train")
        dataset = dataset.map(self.apply_prompt_template,
                              remove_columns=list(dataset.features))
        dataset = dataset.map(
            lambda sample: self.tokenizer(sample["text"]),
            batched=True,
            remove_columns=list(dataset.features),
        ).map(Concatenator(self.model.config.n_ctx), batched=True)
        dataset = dataset.train_test_split(test_size=0.2)

        return dataset["train"], dataset["test"]

    def get_data_collator(self):
        """
        Load the data collator that is responsible for taking in batches of examples
        and converting them into a format that can be consumed by the model.
        """
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        return data_collator

    def get_compute_metrics(self):
        """
        Load the compute_metrics function that is used by the HuggingFace Trainer to
        evaluate the model.
        :return:
        """
        return self.compute_metrics

    def apply_prompt_template(self, example):
        return {
            "text": self.prompt.format(
                dialog=example["dialogue"],
                summary=example["summary"],
                eos_token=self.tokenizer.eos_token,
            )
        }

    def preprocess(self, examples):
        inputs = self.tokenizer(examples["prompt"],
                                padding=True,
                                truncation=True,)
        summaries = self.tokenizer(examples["summary"],
                                   padding=True,
                                   truncation=True,
                                   )
        # inputs["labels"] = summaries["input_ids"]
        return inputs

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred

        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        results = self.metrics.compute(predictions=predictions,
                                       references=labels)

        rouge2 = results["rouge2"].mid.fmeasure
        rougeL = results["rougeL"].mid.fmeasure

        return {"rouge2": rouge2, "rougeL": rougeL}


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config = GPT2Config.from_pretrained("gpt2")
    dataset = Samsum(tokenizer, GPT2Model.from_pretrained("gpt2",
                                                          config=config))
    dataset.get_tokenized_dataset()