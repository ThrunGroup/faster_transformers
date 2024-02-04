import os

os.environ["MASTER_PORT"] = "29501"
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
    MistralForCausalLM,
    MistralConfig,
)
import copy
from transformers.integrations.deepspeed import deepspeed_init
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_dataset, load_metric
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from evaluate import load
from transformers.utils import is_sagemaker_mp_enabled, is_sagemaker_dp_enabled
import logging
import torch
import os
import gc
import wandb
import numpy as np
import sys
import argparse

from experiments.data.get_dataset import get_dataset
from experiments.models.sparse_mistral.relufication import (
    SparseSiLUTrainer,
    enable_sparse_silu,
)
from experiments.models.sparse_mistral.sparsification_sftt import (
    SparseSFTTTrainer,
    SparseTrainer,
    apply_mistral_sparse_silu_mlp,
    activate_stats,
    enable_sparse_silu,
    enable_relu,
    print_dead_neuron_stats,
    set_sparse_threshold,
    plot_act,
    deactivate_stats,
    load_act_hist,
    save_act_hist,
    GradualSparsificationScheduler,
)
from utils.mistral_utils import compress_mistral
from utils.utils import (
    print_size_of_model,
    is_running_deepspeed,
    is_mainprocess,
    get_datetime,
)
from utils.constants import (
    MISTRAL_YES_ID,
    MISTRAL_NO_ID,
    OPENWEBTEXT,
    MISTRAL_7B,
)


def load_glue(data_type: str, seed: int = 0):
    # Load = dataset
    if data_type == "cola":
        dataset = load_dataset("glue", data_type)
    elif data_type == "boolq":
        dataset = load_dataset("super_glue", data_type)
    else:
        raise NotImplementedError(f"{data_type} is not implemented.")

    # Split the training dataset into 85% train and 15% new validation
    train_dataset = dataset["train"].train_test_split(
        test_size=0.15, seed=seed
    )

    # Set the new splits
    dataset["train"] = train_dataset["train"]
    dataset["validation"] = train_dataset["test"]
    dataset["test"] = dataset["validation"]  # Set original validation
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


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]

    # Extract values at these positions
    values_at_yes = logits[..., MISTRAL_YES_ID]
    values_at_no = logits[..., MISTRAL_NO_ID]

    # Compare and get the index where the value is larger
    # If the value at c1 is greater, return c1; otherwise, return c2
    max_value_indices = torch.where(
        values_at_yes > values_at_no, MISTRAL_YES_ID, MISTRAL_NO_ID
    )

    return max_value_indices


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    preds, labels = extract_response(preds, labels)

    accuracy_metric = load("accuracy")
    results = accuracy_metric.compute(predictions=preds, references=labels)

    return results


def train(exp_config, use_wandb: bool = True, use_sweep: bool = False):
    if use_wandb:
        if is_running_deepspeed():
            if int(os.environ["LOCAL_RANK"]) == 0:
                wandb_run = wandb.init()
            if use_sweep:
                exp_config = wandb.config
            else:
                wandb.config = exp_config
        else:
            wandb_run = wandb.init()
            if use_sweep:
                exp_config = wandb.config
            else:
                wandb.config = exp_config

    model_name = exp_config.model_name
    max_steps = exp_config.max_steps
    num_epochs = exp_config.num_epochs
    train_batch_size = exp_config.train_batch_size
    test_batch_size = exp_config.test_batch_size
    gradient_checkpointing = exp_config.gradient_checkpointing
    use_gradual_sparsification = exp_config.use_gradual_sparsification
    num_warmup_steps = exp_config.num_warmup_steps
    increment_ratio = exp_config.increment_ratio
    push_to_hub = exp_config.push_to_hub
    is_debugging = exp_config.is_debugging
    dataset_type = exp_config.dataset_type
    is_plot = exp_config.is_plot
    seed = exp_config.seed
    use_sparse_model = exp_config.use_sparse_model
    use_relu = exp_config.use_relu
    do_pretraining = exp_config.do_pretraining

    if not use_sparse_model:
        exp_config.use_sparse_regularization = exp_config.print_sparsity = (
            exp_config.print_act_stats
        ) = exp_config.set_sparsity_aware_threshold = False

    if use_wandb:
        run_name = f"{model_name}_{dataset_type}"
        if use_sparse_model:
            run_name += f"_{exp_config.targeted_sparsity}"
        if is_debugging:
            run_name += "_debugging"
        if use_gradual_sparsification:
            run_name += "_warmup"
        if use_relu:
            run_name += "_relu"
        run_name += "_" + get_datetime(True)
        wandb.name = run_name

        if is_running_deepspeed():
            if is_mainprocess():
                wandb_run = wandb.init(
                    project=f"{model_name}_{dataset_type}",
                    name=run_name,
                    reinit=True,
                    config=exp_config,
                )
        else:
            wandb_run = wandb.init(
                project=f"{model_name}_{dataset_type}",
                name=run_name,
                reinit=True,
                config=exp_config,
            )

    debugging_output_dir = ""

    model, tokenizer = prepare_sparse_model(
        is_debugging,
        use_sparse_model,
        exp_config.use_sparse_regularization,
        use_relu,
    )

    print("Loading a dataset...")
    if do_pretraining:
        dataset = get_dataset(OPENWEBTEXT, tokenizer, MISTRAL_7B)
        train_dataset, val_dataset, test_dataset = (
            dataset.get_tokenized_dataset()
        )
        sample_dataset = test_dataset
        max_seq_length = 128
        data_collator = dataset.get_data_collator()
        compute_metrics = None
        preprocess_logits_for_metrics = None
        trainer_func = SparseTrainer
    else:
        if dataset_type == "cola":
            dataset = load_glue(dataset_type, seed=seed)
            formatting_func = formatting_func_cola
            create_prompt = create_cola_prompt
        elif dataset_type == "boolq":
            dataset = load_glue(dataset_type, seed=seed)
            formatting_func = formatting_func_boolq
            create_prompt = create_boolq_prompt
        max_seq_length = 1024

        def tokenize(element):
            outputs = tokenizer(
                formatting_func(element),
                add_special_tokens=True,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )
            return {
                "input_ids": outputs["input_ids"],
                "attention_mask": outputs["attention_mask"],
            }

        # Roundabout way to avoid error
        test_dataset = dataset["test"].map(
            tokenize,
            batched=True,
            remove_columns=dataset["test"].column_names,
            num_proc=None,
            batch_size=test_batch_size,
        )
        if exp_config.set_sparsity_aware_threshold:
            sample_dataset = Dataset.from_dict(dataset["train"][:1000])
            sample_dataset = sample_dataset.map(
                tokenize,
                batched=True,
                remove_columns=sample_dataset.column_names,
                num_proc=None,
                batch_size=train_batch_size,
            )
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]

        response_template = "Response:"
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template, tokenizer=tokenizer
        )
        trainer_func = SparseSFTTTrainer

    if is_debugging:
        debugging_output_dir = "debugging_output"
        os.makedirs(debugging_output_dir, exist_ok=True)
        train_dataset = train_dataset.select(range(300))
        val_dataset = val_dataset.select(range(300))
        test_dataset = test_dataset.select(range(300))
        output_dir = debugging_output_dir
    else:
        output_dir = os.path.join(exp_config.output_dir, model_name)

    trainer_config = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=50,  # early stopping counts only when eval step and save step match
        save_steps=50,
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
        gradient_accumulation_steps=2,
        deepspeed="mistral_ds_config.json" if not is_debugging else None,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=test_batch_size,
        gradient_checkpointing=gradient_checkpointing,
        bf16=True if not is_debugging else False,
        ddp_find_unused_parameters=True,
        hub_model_id=f"thrunlab/{model_name}",
        push_to_hub=push_to_hub,
        seed=seed,
        data_seed=seed,
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=6)
    sparsification_scheduler = GradualSparsificationScheduler(
        num_warmup_steps=num_warmup_steps,
        increment_ratio=increment_ratio,
        is_enabled=use_gradual_sparsification,
    )

    trainer_args = {
        "model": model,
        "tokenizer": tokenizer,
        "data_collator": data_collator,
        "args": trainer_config,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "callbacks": [early_stopping, sparsification_scheduler],
        "use_sparse_regularization": exp_config.use_sparse_regularization,
    }

    if not do_pretraining:
        trainer_args["formatting_func"] = formatting_func
        trainer_args["max_seq_length"] = max_seq_length
        trainer_args["compute_metrics"] = compute_metrics
        trainer_args["preprocess_logits_for_metrics"] = (
            preprocess_logits_for_metrics
        )

    trainer = trainer_func(**trainer_args)

    # Collect statistics and find the threshold for given sparsity
    if exp_config.set_sparsity_aware_threshold:
        print("Setting Sparsity-Aware Threshold...")
        if is_debugging:
            # act_hist_path = "/tmp/pre_finetune__act_hist.pt"
            act_hist_path = (
                f"histograms/debugging/{model_name}/{dataset_type}_act_hist.pt"
            )
        else:
            # act_hist_path = f"/scr/jay/exps/pre_finetune/{model_name}/{dataset_type}_act_hist.pt"
            act_hist_path = (
                f"histograms/{model_name}/{dataset_type}_act_hist.pt"
            )

        activate_stats(model.get_base_model())
        if False and os.path.exists(act_hist_path):  # @luke TODO: remove False
            load_act_hist(model.get_base_model(), act_hist_path)
        else:
            is_deepspeed_enabled = trainer.is_deepspeed_enabled
            # trainer.is_deepspeed_enabled = False  # A way to go around the value error when using ds stage 2 for evaluation
            # trainer.evaluate(sample_dataset)
            trainer_config2 = copy.deepcopy(trainer_config)
            trainer_config2.max_steps = 1
            trainer_args2 = copy.deepcopy(trainer_args)
            trainer_args2["args"] = trainer_config2
            trainer2 = trainer_func(**trainer_args2)
            trainer2.train()
            trainer2.evaluate(test_dataset)
            # trainer.is_deepspeed_enabled = is_deepspeed_enabled
            save_act_hist(model.get_base_model(), act_hist_path)

        set_sparse_threshold(
            model.get_base_model(), exp_config.targeted_sparsity
        )

    # Plot activation histograms
    if is_plot:
        if is_running_deepspeed():
            if is_mainprocess():
                plot_act(model.get_base_model())
        else:
            plot_act(model.get_base_model())

    # Train Model after deactivating collecting statistics
    deactivate_stats(model.get_base_model())
    print("Training...")
    trainer.train()

    # Evaluate on test dataset
    model = model.merge_and_unload()

    if exp_config.print_sparsity:
        activate_stats(model, exp_config.print_act_stats)
    eval_result = trainer.evaluate(test_dataset)
    print("===Test Result==")
    print(eval_result)
    log_dict = {
        "test_accuracy": eval_result["eval_accuracy"],
    }

    if exp_config.print_sparsity:
        total_sparsity = print_dead_neuron_stats(model)
        log_dict.update({"total_sparsity": total_sparsity})

    if exp_config.print_act_stats:
        if is_debugging:
            # act_hist_path = "/tmp/post_finetune__act_hist.pt"
            act_hist_path = (
                f"histograms/debugging/{model_name}/{dataset_type}_act_hist.pt"
            )
        else:
            # act_hist_path = "/scr/jay/exps/post_finetune/{model_name}/{dataset_type}_act_hist.pt"
            act_hist_path = (
                f"histograms/{model_name}/{dataset_type}_act_hist.pt"
            )
        save_act_hist(model, act_hist_path)
        plot_act(model)

    if use_wandb:
        if is_running_deepspeed():
            if is_mainprocess():
                wandb.log(log_dict)
        else:
            wandb.log(log_dict)

    if exp_config.model_save:
        trainer.save_model()

    # Empty out gpu memory
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def prepare_sparse_model(
    is_debugging=False,
    use_sparse_model: bool = True,
    use_sparse_regularization: bool = False,
    use_relu: bool = False,
):
    print("Loading a model...")
    if is_debugging:
        config = MistralConfig(
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            # num_attention_heads=4,
        )
        model = MistralForCausalLM(config=config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
        )

    if use_sparse_model:
        print("Enabling Sparse SiLU...")
        apply_mistral_sparse_silu_mlp(
            model,
            model.config,
            use_sparse_regularization=use_sparse_regularization,
        )
        if use_relu:
            enable_relu(model)
        else:
            enable_sparse_silu(model)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Applying LoRA...")
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for text classification"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random Seed",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Mistral_Sparse_0.1",
        help="Name of the model",
    )
    parser.add_argument(
        "--dataset_type", type=str, default="cola", help="Dataset Type"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        # default="/scr/jay/ckpt",
        default="ckpt",
        help="Output Directory",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=8, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Max number of training steps",
    )
    parser.add_argument(
        "--do_pretraining",
        action="store_true",
        default=False,
        help="Whether to pretrain the model on OpenWebText",
    )
    parser.add_argument(
        "--push_to_hub",
        type=bool,
        default=False,
        help="Whether to push the model to Hugging Face Hub",
    )
    parser.add_argument("--model_save", type=bool, default=False)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Training batch size"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=32, help="Testing batch size"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=False,
        help="Use gradient checkpointing",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--is_debugging", action="store_true", default=False)
    parser.add_argument("--is_plot", action="store_true", default=False)
    parser.add_argument(
        "--set_sparsity_aware_threshold",
        action="store_true",
        default=False,
        help="Set sparsity aware threshold",
    )
    parser.add_argument(
        "--use_gradual_sparsification",
        action="store_true",
        default=False,
        help="Whether to do gradually increase the threshold for killing neurons",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=40,
        help="Number of training steps required to reach the dead threshold",
    )
    parser.add_argument(
        "--increment_ratio",
        type=float,
        default=0.5,
        help="By how much to increase the dead threshold",
    )
    parser.add_argument(
        "--print_act_stats",
        action="store_true",
        default=False,
        help="Print pre/post-activation statistics.",
    )
    parser.add_argument(
        "--print_sparsity",
        action="store_true",
        default=False,
        help="Print Sparsity of the mlp layers for test dataset.",
    )
    parser.add_argument(
        "--targeted_sparsity",
        type=float,
        default=0.9,
        help="Targeted Sparsity of the sparse mlp layers in Mistral.",
    )

    parser.add_argument(
        "--use_sparse_model",
        action="store_true",
        default=False,
        help="Whether to use sparse Mistral model",
    )
    parser.add_argument(
        "--use_sparse_regularization",
        action="store_true",
        default=False,
        help="Whether to use sparse regularization",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Whether to log in wandb",
    )
    parser.add_argument(
        "--use_relu",
        action="store_true",
        default=False,
        help="Whether to replace SiLU with ReLU",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    train(args, use_wandb=args.use_wandb)
