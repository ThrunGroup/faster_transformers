from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
import torch
from datasets import load_dataset, load_metric, Dataset
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from evaluate import load
from typing import List
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

import numpy as np


def load_cola():
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
            index = np.where(row == a)[0][
                0
            ]  # Find the first occurrence of 'a'
            if index > 0:
                result.append(row[index - 1])
        except IndexError:
            # This occurs if 'a' is not found in the row
            print(
                "ERROR: No EOS Token is found in the prompt when evaluating."
            )
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
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
            ]:
                new_logits_row.append(logits_row[idx - 1])
                new_labels_row.append(labels_row[idx])
        if len(new_logits_row) > 1:
            print(new_logits_row, new_labels_row)
        if len(new_logits_row) == 0:
            print(new_logits_row, new_labels_row)
        if len(new_logits_row) > 0:
            new_logits.append(new_logits_row)
            new_labels.append(new_labels_row)

    return np.array(new_logits, dtype=int), np.array(new_labels, dtype=int)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    print("NOTHING PLEASE!!!")
    # print(type(preds), type(labels))

    if isinstance(preds, tuple):
        preds = preds[0]
    # print(preds.shape, labels.shape)
    # # Debugging
    # print("PRINTING FIRST ELEMENT of preds and labels. ALSO Printing decoded elements.")
    # print(preds[0], labels[0])
    # labels = [[token if token != -100 else tokenizer.pad_token_id for token in sequence] for sequence in labels]
    # preds = [[token if token != -100 else tokenizer.pad_token_id for token in sequence] for sequence in preds]
    # print(tokenizer.batch_decode([preds[0]]))
    # print(tokenizer.batch_decode([labels[0]]))

    # # Replace -100 in the preds as we can't decode them
    # preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    # # Decode generated summaries into text
    # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # # Replace -100 in the labels as we can't decode them
    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # # Decode reference summaries into text
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # # ROUGE expects a newline after each sentence
    # decoded_preds = ["\n".join(pred.strip()) for pred in decoded_preds]

    # decoded_labels = ["\n".join(label.strip()) for label in decoded_labels]

    # print(decoded_preds[0], decoded_labels[0])

    preds, labels = extract_response(preds, labels)
    # print(preds, labels)

    exact_match_metric = load("accuracy")
    results = exact_match_metric.compute(predictions=preds, references=labels)

    return results


peft_config = LoraConfig(
    lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM"
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

dataset_type = "boolq"
if dataset_type == "cola":
    dataset = load_cola()
    formatting_func = formatting_func_cola
    create_prompt = create_cola_prompt
elif dataset_type == "boolq":
    dataset = load_boolq()
    formatting_func = formatting_func_boolq
    create_prompt = create_boolq_prompt
print(create_prompt(dataset["train"][0]))

response_template = "Response: "
collator = DataCollatorForCompletionOnlyLM(
    response_template, tokenizer=tokenizer
)

args = TrainingArguments(
    output_dir="mistral_instruct_generation",
    num_train_epochs=5,
    # max_steps = 50, # comment out this line if you want to train in epochs
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=0.03,
    logging_steps=10,
    save_strategy="epoch",
    # evaluation_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=1,  # comment out this line if you want to evaluate at the end of each epoch
    learning_rate=2e-4,
    bf16=True,
    lr_scheduler_type="constant",
)

max_seq_length = 2048
trainer = SFTTrainer(
    model=model,
    # peft_config=peft_config,
    # max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    data_collator=collator,
    formatting_func=formatting_func,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()
trainer.evaluate()
