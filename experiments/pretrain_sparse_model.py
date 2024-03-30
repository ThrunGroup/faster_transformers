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

from mistral_instruct_tuning import *


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
    num_epochs = exp_config.num_epochs
    train_batch_size = exp_config.train_batch_size
    test_batch_size = exp_config.test_batch_size
    gradient_checkpointing = exp_config.gradient_checkpointing
    push_to_hub = exp_config.push_to_hub
    is_debugging = exp_config.is_debugging
    dataset_type = exp_config.dataset_type
    is_plot = exp_config.is_plot
    seed = exp_config.seed

    # Gradual Sparsification
    use_gradual_sparsification = exp_config.use_gradual_sparsification
    num_warmup_steps = exp_config.num_warmup_steps
    increment_ratio = exp_config.increment_ratio

    # Graceful Regularization
    use_graceful_regularization = exp_config.use_graceful_regularization
    keep_regularization_with_kill = exp_config.keep_regularization_with_kill

    # If not using sparse Mistral model, all flags related to sparse model should be set as zero or false.
    if not exp_config.use_sparse_model:
        exp_config.use_sparse_regularization = (
            exp_config.print_sparsity
        ) = exp_config.print_act_stats = exp_config.set_sparsity_aware_threshold = False
        exp_config.targeted_sparsity = 0
        # exp_config.sparse_model_dir = None

    if exp_config.use_relu:
        exp_config.targeted_sparsity = "ReLU"
    elif exp_config.use_resilu:
        exp_config.targeted_sparsity = "ReSiLU"

    run_name = f"{model_name}_{dataset_type}"
    if exp_config.use_relu:
        run_name += "_relu"
    elif exp_config.use_resilu:
        run_name += "_resilu"
    else:
        run_name += f"_{int(exp_config.targeted_sparsity * 100)}p"

    if exp_config.cut_pre_mlp:
        run_name += "_cut_pre_mlp"
    if exp_config.cut_pre_attn:
        run_name += "_cut_pre_attn"

    # Load model
    model, tokenizer, config = prepare_sparse_model(
        is_debugging,
        exp_config.use_sparse_model,
        exp_config.use_sparse_regularization,
        exp_config.use_spm,
        exp_config.sparse_model_dir,
        exp_config.use_lora,
        exp_config.use_flash_attn,
        base_model_name=MISTRAL_7B,
        gradient_checkpointing=gradient_checkpointing,
        use_relu=exp_config.use_relu,
        use_resilu=exp_config.use_resilu,
        cut_pre_mlp=exp_config.cut_pre_mlp,
        cut_pre_attn=exp_config.cut_pre_attn,
    )

    # Load dataset
    dataset = get_dataset(
        dataset_type, tokenizer, MISTRAL_7B, max_seq_length=exp_config.max_seq_length
    )
    train_dataset, val_dataset, test_dataset = dataset.get_tokenized_dataset()
    print(train_dataset, val_dataset, test_dataset)
    data_collator = dataset.get_data_collator()

    if is_debugging:
        output_dir = "debugging_output"
        # dataset["train"] = dataset["train"].select(range(300))
        # dataset["validation"] = dataset["validation"].select(range(300))
        # dataset["test"] = dataset["test"].select(range(300))
    else:
        output_dir = os.path.join(exp_config.output_dir, run_name)

    # Use only 500 samples of training dataset to collect statistics
    sampled_train_dataset = train_dataset.select(range(1000))

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

    trainer_config = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=25,  # early stopping counts only when eval step and save step match
        max_steps=10 if is_debugging else exp_config.max_steps,
        save_steps=min(250, exp_config.max_steps),
        logging_steps=5,
        save_strategy="steps",
        learning_rate=1e-5,
        weight_decay=0.01,
        num_train_epochs=num_epochs,
        logging_dir="exp_logs",
        save_total_limit=11,
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        greater_is_better=False,
        # report_to="wandb" if use_wandb else None,
        gradient_accumulation_steps=exp_config.gradient_accumulation_steps,
        deepspeed=exp_config.ds_config_path if not is_debugging else None,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=test_batch_size,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True if not is_debugging else False,
        # ddp_find_unused_parameters=True,
        hub_model_id=f"thrunlab/{run_name}",
        push_to_hub=push_to_hub,
        seed=seed,
        data_seed=seed,
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=1000, early_stopping_threshold=0.002
    )
    sparsification_scheduler = GradualSparsificationScheduler(
        num_warmup_steps=num_warmup_steps,
        increment_ratio=increment_ratio,
        is_enabled=use_gradual_sparsification,
        model_name=model_name,
    )
    graceful_regularization_scheduler = GracefulRegularizationScheduler(
        num_warmup_steps=num_warmup_steps,
        is_enabled=use_graceful_regularization,
        model_name=model_name,
        test_dataset=test_dataset,
        targeted_sparsity=exp_config.targeted_sparsity,
        keep_regularization_with_kill=keep_regularization_with_kill,
    )

    # See https://discuss.huggingface.co/t/using-iterabledataset-with-trainer-iterabledataset-has-no-len/15790/2
    trainer = SparseTrainer(
        model=model,
        # peft_config=peft_config,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=trainer_config,
        train_dataset=train_dataset.with_format(
            "torch"
        ),  # train_dataset.with_format("torch"),
        eval_dataset=val_dataset.with_format("torch"),
        # compute_metrics=dataset.compute_metrics,
        # preprocess_logits_for_metrics=dataset.preprocess_logits_for_metrics,
        callbacks=[],
        use_sparse_regularization=exp_config.use_sparse_regularization,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        config.use_cache = False
        model.config.use_cache = False
        # model.enable_input_require_grads()  # See https://github.com/huggingface/transformers/issues/23170

    if exp_config.use_lora:
        base_model = model.get_base_model()
    else:
        base_model = model

    # Collect statistics and find the threshold for given sparsity
    if not use_graceful_regularization and exp_config.set_sparsity_aware_threshold:
        if is_debugging:
            act_hist_path = "/tmp/pre_finetune__act_hist"
        else:
            act_hist_path = (
                f"/scr/jay/exps/pre_finetune/{model_name}/{dataset_type}_act_hist"
            )

            if exp_config.cut_pre_mlp:
                act_hist_path += "_cut_pre_mlp"
            if exp_config.cut_pre_attn:
                act_hist_path += "_cut_pre_attn"

        activate_stats(base_model)
        if os.path.exists(act_hist_path):
            load_act_hist(base_model, act_hist_path)
        elif not exp_config.use_relu:
            print("Collecting Statistics...")
            is_deepspeed_enabled = trainer.is_deepspeed_enabled
            trainer.is_deepspeed_enabled = False  # A way to go around the value error when using ds stage 2 for evaluation
            trainer.evaluate(sampled_train_dataset)
            trainer.is_deepspeed_enabled = is_deepspeed_enabled
            save_act_hist(base_model, act_hist_path)

        if exp_config.use_relu:
            set_sparse_threshold(base_model, 0, use_relu=True)
        elif exp_config.use_resilu:
            set_sparse_threshold(base_model, 0, use_resilu=True)
        else:
            set_sparse_threshold(base_model, exp_config.targeted_sparsity)

    if exp_config.print_sparsity:
        if exp_config.use_lora:
            base_model = model.get_base_model()
        else:
            base_model = model
        activate_stats(base_model, exp_config.print_act_stats)

        is_deepspeed_enabled = trainer.is_deepspeed_enabled
        trainer.is_deepspeed_enabled = False
        eval_result = trainer.evaluate(test_dataset)
        trainer.is_deepspeed_enabled = is_deepspeed_enabled

        print("===Pre-training test Result==")
        print(eval_result)
        total_sparsity = print_dead_neuron_stats(base_model)
        print(f"pre-training sparsity: {total_sparsity}")

        # if exp_config.set_sparsity_aware_threshold:
        #     if exp_config.use_relu:
        #         set_sparse_threshold(base_model, 0, use_relu=True)
        #     elif exp_config.use_resilu:
        #         set_sparse_threshold(base_model, 0, use_resilu=True)
        #     else:
        #         set_sparse_threshold(base_model, exp_config.targeted_sparsity)

    # Plot activation histograms
    if is_plot:
        fig_dir = f"figures/{run_name}"
        if is_running_deepspeed():
            if is_mainprocess():
                plot_act(base_model, fig_dir=fig_dir)
        else:
            plot_act(base_model, fig_dir=fig_dir)

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
            warnings.warn(
                "use_spm arg is ignored as use_spare_model arg is not activated."
            )
    trainer.train(resume_from_checkpoint=exp_config.resume_from_checkpoint)

    # Evaluate on test dataset
    if exp_config.use_lora:
        model = model.merge_and_unload()

    # Enable sparse predictor in inference stage
    if exp_config.use_spm:
        if exp_config.use_sparse_model:
            enable_sparse_predictor(model)
        else:
            warnings.warn(
                "use_spm arg is ignored as use_spare_model arg is not activated."
            )

    if exp_config.print_sparsity:
        activate_stats(model, exp_config.print_act_stats)
    eval_result = trainer.evaluate(test_dataset)
    print("===Test Result==")
    print(eval_result)
    log_dict = {}

    if exp_config.print_sparsity:
        total_sparsity = print_dead_neuron_stats(model)
        log_dict.update({"total_sparsity": total_sparsity})

    if exp_config.print_act_stats:
        if is_debugging:
            act_hist_path = "/tmp/post_finetune__act_hist"
        else:
            act_hist_path = (
                f"/scr/jay/exps/post_finetune/{model_name}/{dataset_type}_act_hist"
            )

        if exp_config.cut_pre_mlp:
            act_hist_path += "_cut_pre_mlp"
        if exp_config.cut_pre_attn:
            act_hist_path += "_cut_pre_attn"

        save_act_hist(model, act_hist_path)
        plot_act(model)

    if exp_config.model_save:
        # Save thresholds
        if exp_config.use_sparse_model:
            thresholds = [float(m.mlp.dead_threshold) for m in model.model.layers]
            model.config.thresholds = thresholds

            if exp_config.cut_pre_mlp:
                pre_mlp_thresholds = [
                    float(m.mlp.pre_mlp_threshold) for m in model.model.layers
                ]
                model.config.pre_mlp_thresholds = pre_mlp_thresholds

            if exp_config.cut_pre_attn:
                pre_attn_thresholds = [
                    float(m.self_attn.pre_attn_threshold) for m in model.model.layers
                ]
                model.config.pre_attn_thresholds = pre_attn_thresholds

        # model.config.save_pretrained(output_dir)
        model.save_pretrained(output_dir + f"_no_adapter_steps_{exp_config.max_steps}")
        tokenizer.save_pretrained(output_dir + f"_no_adapter_steps_{exp_config.max_steps}")

    if push_to_hub:
        trainer.model = model
        trainer.deepspeed = model
        trainer.push_to_hub()

    # Early Exit/ Early Entry experiments
    # model.model.original_layers = nn.ModuleList([layer for layer in model.model.layers])
    # for idx in range(len(model.model.original_layers)):
    #     enable_last_k_modules(model, idx)
    #     print(f"Early Start: layer {idx}!!!!")
    #     eval_result = trainer.evaluate(test_dataset)
    #     print(eval_result)
    #     log_dict[f"early start-{idx} accuracy"] = eval_result["eval_accuracy"]
    #
    # for idx in range(len(model.model.original_layers)):
    #     enable_first_k_modules(model, idx)
    #     print(f"Early Exit: layer {idx}!!!!")
    #     eval_result = trainer.evaluate(test_dataset)
    #     print(eval_result)
    #     log_dict[f"early exit-{idx} accuracy"] = eval_result["eval_accuracy"]

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
