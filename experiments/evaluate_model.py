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
from transformers.integrations.deepspeed import deepspeed_init
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_dataset, load_metric
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from evaluate import load
from transformers.utils import is_sagemaker_mp_enabled, is_sagemaker_dp_enabled
from deepspeed.runtime.zero.stage_1_and_2 import (
    estimate_zero2_model_states_mem_needs_all_live,
)


import torch
import deepspeed
import torch.nn as nn
import os
import gc
import wandb
import numpy as np
import sys
import argparse
import time
import warnings

from experiments.models.sparse_mistral.sparsification_sftt import (
    SparseSFTTTrainer,
    SparseTrainer,
    SparseMistralforCausalLM,
    GradualSparsificationScheduler,
    GracefulRegularizationScheduler,
    apply_mistral_sparse_silu_mlp,
    apply_mistral_sparse_decoder_layer,
    activate_stats,
    enable_sparse_silu,
    print_dead_neuron_stats,
    set_sparse_threshold,
    plot_act,
    deactivate_stats,
    load_act_hist,
    save_act_hist,
    enable_last_k_modules,
    enable_first_k_modules,
    enable_sparse_predictor,
    disable_sparse_predictor,
    get_sparse_mistral_config,
)
from experiments.models.sparse_mistral.svd_router import SparsePredictor
from utils.mistral_utils import compress_mistral
from utils.utils import (
    print_size_of_model,
    is_running_deepspeed,
    is_mainprocess,
    get_datetime,
    ds_print,
)
from utils.constants import (
    MISTRAL_YES_ID,
    MISTRAL_NO_ID,
    MISTRAL_7B,
    OPENWEBTEXT,
    REFINED_WEB,
    COLA,
    BOOLQ,
    QNLI,
    SST2,
    WIC,
)
from format_dataset_for_instruction_tuning import (
    get_formatting_func,
    get_dataset_for_instruction_tuning,
)
from experiments.data.get_dataset import get_dataset
from format_dataset_for_instruction_tuning import (
    get_formatting_func,
    get_dataset_for_instruction_tuning,
)
from mistral_instruct_tuning import *


def extract_response(logits, labels):
    # ds_print("extract_response logits shape: ", logits.shape)
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
            ]:
                new_logits_row.append(logits_row[idx - 1])
                new_labels_row.append(labels_row[idx])

        if len(new_logits_row) > 0:
            # ds_print(" extract_response new_logits_row: ", np.array(new_logits_row).shape)
            new_logits.append(new_logits_row)
            new_labels.append(new_labels_row)

    return np.array(new_logits, dtype=int), np.array(new_labels, dtype=int)


def classification_preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]

    # Extract values at these positions
    values_at_yes = logits[..., MISTRAL_YES_ID]
    values_at_no = logits[..., MISTRAL_NO_ID]

    # Compare and get the index where the value is larger
    # If the value at c1 is greater, return c1; otherwise, return c2
    max_value_indices = torch.where(values_at_yes > values_at_no, MISTRAL_YES_ID, MISTRAL_NO_ID)

    return max_value_indices


def classification_compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    #
    # ds_print("preds shape: ", preds.shape)
    # ds_print("preds: ", preds)
    # ds_print("label shape: ", labels.shape)
    # ds_print("labels: ", labels)

    preds, labels = extract_response(preds, labels)

    accuracy_metric = load("accuracy")
    results = accuracy_metric.compute(predictions=preds, references=labels)

    return results


def evaluate_zero_shot(
    trainer,
    dataset_types: list = None,
    tokenizer=None,
    train_batch_size=32,
    test_batch_size=32,
    max_seq_length=1024,
    return_sample_dataset=False,
):
    ds_print("Datasets: ", dataset_types)
    ds_print("==Test set evaluation==")
    for dataset_type in dataset_types:
        ds_print(f"Evaluating on {dataset_type} dataset...")
        formatting_func = get_formatting_func(dataset_type)
        eval_accuracy_list = []
        for seed in range(3):
            _, _, test_dataset, _ = get_dataset_for_instruction_tuning(
                dataset_type,
                seed=seed,
                tokenizer=tokenizer,
                formatting_func=formatting_func,
                max_seq_length=max_seq_length,
                train_batch_size=train_batch_size,
                test_batch_size=test_batch_size,
                return_sample_dataset=return_sample_dataset,
            )
            eval_result = trainer.evaluate(test_dataset.select(range(2)))
            print(eval_result)
            eval_accuracy_list += (eval_result["eval_accuracy"],)

        if is_mainprocess():
            ds_print(f"Mean accuracy: {np.mean(eval_accuracy_list)}")


def train(exp_config, use_wandb: bool = True, use_sweep: bool = False):
    print(f"Config: {exp_config}")
    if use_wandb:
        if is_running_deepspeed():
            if int(os.environ["LOCAL_RANK"]) == 0:
                wandb.init()
            if use_sweep:
                exp_config = wandb.config
            else:
                wandb.config = exp_config
            time.sleep(15)
        else:
            wandb.init()
            if use_sweep:
                exp_config = wandb.config
            else:
                wandb.config = exp_config

    model_name = exp_config.model_name
    train_batch_size = exp_config.train_batch_size
    test_batch_size = exp_config.test_batch_size
    gradient_checkpointing = exp_config.gradient_checkpointing
    is_debugging = exp_config.is_debugging
    dataset_type = exp_config.dataset_type
    is_plot = exp_config.is_plot
    seed = exp_config.seed

    # Zero-shot evaluation
    pretrained_model_to_load = exp_config.pretrained_model_to_load
    end_steps = exp_config.end_steps

    # Graceful Regularization
    use_graceful_regularization = exp_config.use_graceful_regularization

    # If not using sparse Mistral model, all flags related to sparse model should be set as zero or false.
    if not exp_config.use_sparse_model:
        exp_config.use_sparse_regularization = (
            exp_config.print_sparsity
        ) = exp_config.print_act_stats = exp_config.set_sparsity_aware_threshold = False
        exp_config.targeted_sparsity = 0
        # exp_config.sparse_model_dir = None

    if exp_config.use_relu:
        exp_config.targeted_sparsity = "ReLU"

    run_name = f"{model_name}_{dataset_type}_{int(exp_config.targeted_sparsity * 100)}p_graceful_{use_graceful_regularization}"

    # Load model
    model, tokenizer, config = prepare_sparse_model(
        is_debugging,
        exp_config.use_sparse_model,
        exp_config.use_sparse_regularization,
        exp_config.use_spm,
        exp_config.sparse_model_dir,
        exp_config.use_lora,
        exp_config.use_flash_attn,
        base_model_name=pretrained_model_to_load,
        gradient_checkpointing=gradient_checkpointing,
    )

    formatting_func = get_formatting_func(COLA)
    max_seq_length = 1024
    # Load dataset
    (
        train_dataset,
        val_dataset,
        test_dataset,
        sample_dataset,
    ) = get_dataset_for_instruction_tuning(
        COLA,
        seed,
        tokenizer,
        formatting_func,
        max_seq_length,
        train_batch_size,
        test_batch_size,
        return_sample_dataset=exp_config.set_sparsity_aware_threshold,
    )

    if is_debugging:
        output_dir = "debugging_output"
        train_dataset = train_dataset.select(range(4))
        val_dataset = train_dataset
        test_dataset = train_dataset
    else:
        output_dir = os.path.join(exp_config.output_dir, run_name)

    # Use only 1000 samples of training dataset to collect statistics
    if exp_config.set_sparsity_aware_threshold:
        sampled_train_dataset = test_dataset

    if use_wandb:
        if is_debugging:
            run_name += "_debugging"
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

    ds_print("Evaluating without training...")

    trainer_config = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        max_steps=1,
        learning_rate=0,
        deepspeed=exp_config.ds_config_path if not is_debugging else None,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=test_batch_size,
        bf16=True,
        hub_model_id=f"thrunlab/{run_name}",
        seed=seed,
        data_seed=seed,
    )

    # See https://discuss.huggingface.co/t/using-iterabledataset-with-trainer-iterabledataset-has-no-len/15790/2
    response_template = "Response:"
    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SparseSFTTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=trainer_config,
        train_dataset=train_dataset.with_format("torch"),  # train_dataset.with_format("torch"),
        eval_dataset=val_dataset.with_format("torch"),
        use_sparse_regularization=exp_config.use_sparse_regularization,
        formatting_func=formatting_func,
        max_seq_length=1024,
        compute_metrics=classification_compute_metrics,
        preprocess_logits_for_metrics=classification_preprocess_logits_for_metrics,
    )

    if exp_config.use_lora and not is_debugging:
        base_model = model.get_base_model()
    else:
        base_model = model

    # Collect statistics and find the threshold for given sparsity
    if not use_graceful_regularization and exp_config.set_sparsity_aware_threshold:
        if is_debugging:
            act_hist_path = "/tmp/pre_finetune__act_hist.pt"
        else:
            act_hist_path = f"/matx/u/lukeai/histograms/{model_name}_{dataset_type}_act_hist.pt"

        activate_stats(base_model)
        if os.path.exists(act_hist_path):
            load_act_hist(base_model, act_hist_path)
        else:
            is_deepspeed_enabled = trainer.is_deepspeed_enabled
            trainer.is_deepspeed_enabled = False  # A way to go around the value error when using ds stage 2 for evaluation
            trainer.evaluate(sampled_train_dataset)
            trainer.is_deepspeed_enabled = is_deepspeed_enabled
            save_act_hist(base_model, act_hist_path)
        if exp_config.use_relu:
            set_sparse_threshold(base_model, 0, True)
        else:
            set_sparse_threshold(base_model, exp_config.targeted_sparsity)

    # Plot activation histograms
    if is_plot:
        if is_running_deepspeed():
            if is_mainprocess():
                plot_act(base_model)
        else:
            plot_act(base_model)

    # Train Model after deactivating collecting statistics
    deactivate_stats(base_model)

    # Only use distillation loss if training sparse predictor
    if exp_config.use_spm:
        if exp_config.use_sparse_model:
            trainer.freeze_original_weights = False
            trainer.initialize_sparse_decoder_layers(base_model)
            trainer.use_spm_loss = True
            disable_sparse_predictor(base_model)

            is_deepspeed_enabled = trainer.is_deepspeed_enabled
            trainer.is_deepspeed_enabled = False  # A way to go around the value error when using ds stage 2 for evaluation
            print(trainer.evaluate(test_dataset))
            trainer.is_deepspeed_enabled = is_deepspeed_enabled

            enable_sparse_predictor(base_model)
        else:
            warnings.warn("use_spm arg is ignored as use_spare_model arg is not activated.")

    trainer.train(resume_from_checkpoint=exp_config.resume_from_checkpoint)

    # Evaluate on test dataset
    if exp_config.use_lora and not is_debugging:
        model = model.merge_and_unload()

    # Enable sparse predictor in inference stage
    if exp_config.use_spm:
        if exp_config.use_sparse_model:
            enable_sparse_predictor(model)
        else:
            warnings.warn("use_spm arg is ignored as use_spare_model arg is not activated.")

    if exp_config.print_sparsity:
        activate_stats(model, exp_config.print_act_stats)

    dataset_types = exp_config.dataset_types.split(",")
    evaluate_zero_shot(
        trainer,
        dataset_types,
        tokenizer,
        train_batch_size,
        test_batch_size,
        1024,
    )

    if use_wandb:
        if is_running_deepspeed():
            if is_mainprocess():
                wandb.log(log_dict)
        else:
            wandb.log(log_dict)

    # Empty out gpu memory
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # torch.cuda.memory._record_memory_history(max_entries=100000)
    args = parse_args()
    args.dataset_type = REFINED_WEB
    train(args, use_wandb=args.use_wandb)
    # torch.cuda.memory._dump_snapshot("snapshot_new.pickle")
    # deepspeed experiments/pretrain_sparse_model.py --use_sparse_model --targeted_sparsity 0.75 --set_sparsity_aware_threshold --print_sparsity --use_wandb --num_epochs 1 --gradient_checkpointing --model_save --train_batch_size 1 --test_batch_size 1 --use_flash_attn --max_steps 2000 --gradient_accumulation_steps 8
