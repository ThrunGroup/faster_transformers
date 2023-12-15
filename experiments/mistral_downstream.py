from utils.constants import MISTRAL_7B, BOOLQ
from create_models import create_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    AutoPeftModelForSequenceClassification,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

import torch
import argparse
import os

from data.get_dataset import get_dataset


def train_mistral(use_relu: bool = True,):
    dataset_type = BOOLQ  # @jay: fix hard-coded

    os.environ["WANDB_PROJECT"] = f"Mistral 7b on {dataset_type}"
    os.environ["WANDB_WATCH"] = "gradients"
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint"


    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MISTRAL_7B,
        device_map="auto",
        quantization_config=nf4_config,
        use_cache=False,
    )
    if use_relu:
        model.config.hidden_act = "relu"

    tokenizer = AutoTokenizer.from_pretrained(MISTRAL_7B)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    output_dir = f"{MISTRAL_7B}_{dataset_type}"

    args = TrainingArguments(
        output_dir=output_dir,
        # num_train_epochs=5,
        # per_device_train_batch_size=4,
        max_steps=1000,  # comment out this line if you want to train in epochs
        per_device_train_batch_size=4,
        warmup_steps=20,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=20,  # comment out this line if you want to evaluate at the end of each epoch
        learning_rate=2e-4,
        gradient_checkpointing=True,
        # weight_decay=0.01,
        bf16=True,
        lr_scheduler_type="constant",
        hub_model_id=f"thrunlab/{output_dir}",
        push_to_hub=True,
        seed=1,
        data_seed=1,
        report_to="wandb",
        run_name=output_dir,
    )

    # Get dataset
    dataset = get_dataset(dataset_type, tokenizer, model, MISTRAL_7B)
    train_dataset, val_dataset, test_dataset = dataset.get_tokenized_dataset()
    max_seq_length = 131072

    # Call relevant functions
    data_collator = dataset.get_data_collator()
    compute_metrics = dataset.get_compute_metrics()
    preprocess_logits_for_metrics = dataset.preprocess_logits_for_metrics

    trainer = Trainer(
        model=model,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()
    trainer.push_to_hub()

    trainer.evaluate(test_dataset)


def main():
    train_mistral(True)
    train_mistral(False)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default=MISTRAL_7B, help="Model types")
    # parser.add_argument("--dataset", type=str, default=BOOLQ, help="dataset")
    # args = parser.parse_args()
    # common_args = {
    #     "model_type": args.model,
    #     "dataset_name": args.dataset,
    #     "num_epochs": 5,
    #     "train_batch_size": 8,
    #     "test_batch_size": 16,
    #     "gradient_checkpointing": True,
    # }
    # create_model(**common_args, push_to_hub=True,)
