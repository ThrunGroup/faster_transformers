from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self, tokenizer, model, model_type):
        self.tokenizer = tokenizer
        self.model = model
        self.model_type = model_type

    @abstractmethod
    def get_tokenized_dataset(self):
        pass

    @abstractmethod
    def preprocess(self, examples):
        pass

    @abstractmethod
    def compute_metrics(self, logits=None, labels=None):
        pass
