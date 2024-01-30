from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
import pickle
import torch

from .dataset import Dataset


class OpenWebText(Dataset):
    def __init__(self, tokenizer, model, model_type):
        """
        :param tokenizer: Tokenizer to tokenize the inputs
        :param model: Model to finetune / evaluate
        :param metrics: Metrics to evaluate the model
        :param prefix: Prefix to the inputs to start the summarization task
        """
        super().__init__(tokenizer, model, model_type)
        # self.tokenizer.pad_token = tokenizer.mask_toke
        self.tokenizer.pad_token = tokenizer.eos_token

    def get_tokenized_dataset(self):
        """
        Load and build the openwebtext dataset which is already preprocessed.
        :return: dataset
        """
        train_path = (
            f"datasets/{self.model_type}_{self.dataset_type}_train.pkl"
        )
        test_path = f"datasets/{self.model_type}_{self.dataset_type}_test.pkl"
        validation_path = (
            f"datasets/{self.model_type}_{self.dataset_type}_validation.pkl"
        )

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
            print("No dataset found.")
            dataset = load_dataset("Skylion007/openwebtext")

            tokenized_dataset = dataset.map(
                self.preprocess,
                batched=True,
                batch_size=128,
                remove_columns=dataset.column_names["train"],
            )

            print("Tokenization completed.")

            # GLUE doesn't provide answers for TEST, so we have to manually make them
            train_test_dataset = tokenized_dataset["train"].train_test_split(
                test_size=0.1
            )

            train_dataset = train_test_dataset["train"]
            test_dataset = train_test_dataset["test"]
            val_dataset = tokenized_dataset["validation"]

            # Save the preprocessed data
            with open(train_path, "wb") as f:
                pickle.dump(train_dataset, f)
            with open(test_path, "wb") as f:
                pickle.dump(test_dataset, f)
            with open(validation_path, "wb") as f:
                pickle.dump(val_dataset, f)
            print("Preprocessed and saved data to disk.")

        return train_dataset, val_dataset, test_dataset

    def preprocess(self, examples):
        tokenized_inputs = self.tokenizer(
            [" ".join(x) for x in examples["text"]],
            max_length=512,
            truncation=True,
        )

        return tokenized_inputs

    def get_data_collator(self):
        """
        Load the data collator that is responsible for taking in batches of examples
        and converting them into a format that can be consumed by the model.
        """
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, mlm_probability=0.15
        )
        return data_collator

    def compute_metrics(self, preds):
        logits, labels = preds

        # Shift the labels to the right to align with the output of the model
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()

        # Flatten the logits and labels to calculate loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
        )

        # Calculate perplexity
        perplexity = torch.exp(torch.mean(loss))

        return {"perplexity": perplexity.item()}
