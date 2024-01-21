from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_dataset, load_metric
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from evaluate import load

import torch
import os
import gc
import wandb
import numpy as np
import argparse

from experiments.models.sparse_mistral.sparsification_sftt import (
    SparseSFTTTrainer,
    apply_mistral_sparse_silu_mlp,
    activate_stats,
    enable_sparse_silu,
    print_dead_neuron_stats,
)


def load_cola():
    # Load CoLA dataset
    dataset = load_dataset("glue", "cola")
    return dataset


def load_boolq():
    dataset = load_dataset("super_glue", "boolq")
    return dataset


def create_cola_prompt(sample):
    bos_token = "<s>"
    instruction = "Is the following 'input' sentence is grammatically correct? Respond in 'yes' or 'no'."
    input = sample["sentence"]
    response = "yes" if sample["label"] == 1 else "no"
    eos_token = "</s>"

    full_prompt = ""
    full_prompt += bos_token
    full_prompt += f"[INST]### Instruction: {instruction}"
    full_prompt += f"### Input: {input}[/INST]"
    full_prompt += f"### Response: {response}"
    full_prompt += eos_token

    return full_prompt


def create_boolq_prompt(sample):
    bos_token = "<s>"
    instruction = "Based on the following 'passage', determine if the 'question' is true or false. Respond in 'yes' or 'no'."
    passage = sample["passage"]
    question = sample["question"]
    response = "yes" if sample["label"] == 1 else "no"
    eos_token = "</s>"

    full_prompt = ""
    full_prompt += bos_token
    full_prompt += f"[INST]### Instruction: {instruction}"
    full_prompt += f"### Passage: {passage}"
    full_prompt += f"### Question: {question}[/INST]"
    full_prompt += f"### Response: {response}"
    full_prompt += eos_token

    return full_prompt


def formatting_func_cola(samples):
    formatted_prompts = []
    bos_token = "<s>"
    instruction = "Is the following 'input' sentence is grammatically correct? Respond in 'yes' or 'no'."
    eos_token = "</s>"
    for idx in range(len(samples["sentence"])):
        input = samples["sentence"][idx]
        response = "yes" if samples["label"][idx] == 1 else "no"

        full_prompt = ""
        full_prompt += bos_token
        full_prompt += f"[INST]### Instruction: {instruction}"
        full_prompt += f"### Input: {input}[/INST]"
        full_prompt += f"### Response: {response}"
        full_prompt += eos_token
        formatted_prompts.append(full_prompt)
    return formatted_prompts


def formatting_func_boolq(samples):
    formatted_prompts = []
    bos_token = "<s>"
    instruction = "Based on the following 'passage', is the 'question' true or false? Respond in 'yes' or 'no'."
    eos_token = "</s>"

    for idx in range(len(samples["question"])):
        passage = samples["passage"][idx]
        question = samples["question"][idx]
        response = "yes" if samples["label"][idx] == 1 else "no"

        full_prompt = ""
        full_prompt += bos_token
        full_prompt += f"[INST]### Instruction: {instruction}"
        full_prompt += f"### Passage: {passage}"
        full_prompt += f"### Question: {question}[/INST]"
        full_prompt += f"### Response: {response}"
        full_prompt += eos_token
        formatted_prompts.append(full_prompt)

    return formatted_prompts


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def find_left_of_a(np_array, a):
    if not isinstance(np_array, np.ndarray):
        raise ValueError("Input must be a 2D NumPy array.")

    result = []
    for row in np_array:
        if row.size <= 1:
            raise ValueError("Sublist length must be greater than 1.")
        try:
            index = np.where(row == a)[0][0]  # Find the first occurrence of 'a'
            if index > 0:
                result.append(row[index - 1])
        except IndexError:
            # This occurs if 'a' is not found in the row
            print("ERROR: No EOS Token is found in the prompt when evaluating.")
            pass
    return np.array(result)


# https://github.com/huggingface/trl/issues/862
def extract_response(logits, labels):
    # eos_token_id = tokenizer.eos_token_id
    # logits = find_left_of_a(logits, eos_token_id)
    # labels = find_left_of_a(labels, eos_token_id)
    new_logits = []
    new_labels = []
    for row_idx in range(len(labels)):
        new_logits_row = []
        new_labels_row = []
        logits_row = logits[row_idx]
        labels_row = labels[row_idx]

        for idx in range(len(labels[0])):
            if labels_row[idx] not in [
                -100,
                2,
                # tokenizer.eos_token_id,
                # tokenizer.pad_token_id,
            ]:
                new_logits_row.append(logits_row[idx - 1])
                new_labels_row.append(labels_row[idx])
        if len(new_logits_row) > 0:
            new_logits.append(new_logits_row)
            new_labels.append(new_labels_row)

    return np.array(new_logits, dtype=int), np.array(new_labels, dtype=int)


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    preds, labels = extract_response(preds, labels)

    accuracy_metric = load("accuracy")
    results = accuracy_metric.compute(predictions=preds, references=labels)

    return results


def train(args):
    dataset_type = "cola"
    if dataset_type == "cola":
        dataset = load_cola()
        formatting_func = formatting_func_cola
        create_prompt = create_cola_prompt
    elif dataset_type == "boolq":
        dataset = load_boolq()
        formatting_func = formatting_func_boolq
        create_prompt = create_boolq_prompt

    model, tokenizer = prepare_sparse_model()
    print(create_prompt(dataset["train"][0]))

    model_name = args.model_name
    num_epochs = args.num_epochs
    push_to_hub = args.push_to_hub
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    gradient_checkpointing = args.gradient_checkpointing
    push_to_hub = args.push_to_hub

    os.environ["WANDB_PROJECT"] = f"Mistral 7b on {dataset_type}"
    os.environ["WANDB_WATCH"] = "gradients"

    response_template = "Response:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=os.path.join("/scr/jay/ckpt", model_name),
        evaluation_strategy="steps",
        eval_steps=50,  # early stopping counts only when eval step and save step match
        # save_steps=50,
        logging_steps=5,
        save_strategy="steps",
        learning_rate=1e-5,
        weight_decay=0.01,
        num_train_epochs=num_epochs,
        logging_dir="exp_logs",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=model_name,
        gradient_accumulation_steps=2,
        deepspeed="mistral_ds_config.json",
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=test_batch_size,
        gradient_checkpointing=gradient_checkpointing,
        bf16=True,
        ddp_find_unused_parameters=True,
        hub_model_id=f"thrunlab/{model_name}",
        push_to_hub=push_to_hub,
        seed=2,
        data_seed=1,
    )

    max_seq_length = 1024
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
    trainer = SparseSFTTTrainer(
        model=model,
        # peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        data_collator=collator,
        formatting_func=formatting_func,
        args=args,
        train_dataset=Dataset.from_dict(dataset["train"][:]),
        eval_dataset=Dataset.from_dict(dataset["validation"][:]),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[early_stopping],
        use_sparse_regularization=True,
    )
    # activate_stats(model.get_base_model())
    # print(trainer.evaluate())
    # activate_stats(model.get_base_model())
    trainer.train()
    model = model.merge_and_unload()
    activate_stats(model)
    print(trainer.evaluate())
    print_dead_neuron_stats(model)

    # Empty out gpu memory
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def prepare_sparse_model():
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
    )
    apply_mistral_sparse_silu_mlp(model, model.config)
    enable_sparse_silu(model)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for text classification"
    )

    parser.add_argument(
        "--model_name", type=str, default="Mistral_Sparse_0.1", help="Name of the model"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--push_to_hub",
        type=bool,
        default=False,
        help="Whether to push the model to Hugging Face Hub",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Training batch size"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=16, help="Testing batch size"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=False,
        help="Use gradient checkpointing",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    # Add more arguments as needed

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
