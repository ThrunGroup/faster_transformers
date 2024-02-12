import os

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    MistralForCausalLM,
    MistralConfig,
)
from transformers.integrations.deepspeed import deepspeed_init
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_dataset, load_metric
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from evaluate import load
from transformers.utils import is_sagemaker_mp_enabled, is_sagemaker_dp_enabled
import torch
import os
import gc
import wandb
import numpy as np
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
    SparseMistral,
    get_sparse_mistral_config,
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
    MISTRAL_7B,
    OPENWEBTEXT,
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


def classification_preprocess_logits_for_metrics(logits, labels):
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


def classification_compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    preds, labels = extract_response(preds, labels)

    accuracy_metric = load("accuracy")
    results = accuracy_metric.compute(predictions=preds, references=labels)

    return results


def evaluate_zero_shot(trainer: Trainer, dataset_types: list = None,
                       tokenizer=None,
                       train_batch_size=32, test_batch_size=32,
                       max_seq_length=1024,
                       return_sample_dataset=False):
    print("Datasets: ", dataset_types)
    print("==Test set evaluation==")
    for dataset_type in dataset_types:
        print(f"Evaluating on {dataset_type} dataset...")
        formatting_func = get_formatting_func(dataset_type)
        for seed in range(3):
            print("Seed: ", seed)
            _, _, test_dataset, _ = (
                get_dataset_for_instruction_tuning(dataset_type, seed=seed,
                                                   tokenizer=tokenizer,
                                                   formatting_func=formatting_func,
                                                   max_seq_length=max_seq_length,
                                                   train_batch_size=train_batch_size,
                                                   test_batch_size=test_batch_size,
                                                   return_sample_dataset=return_sample_dataset)
            )
            eval_result = trainer.evaluate(test_dataset)
            print(eval_result)


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

    pretrained_model_to_load = exp_config.pretrained_model_to_load
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
    eval_steps = exp_config.eval_steps
    do_evaluate = exp_config.do_evaluate
    learning_rate = 1e-5

    print("==========")
    print(model_name)
    print("==========")

    if do_evaluate:
        print("Evaluating without training...")
        max_steps = 1
        learning_rate = 0.0

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
                    project=f"MISTRAL_{dataset_type}",
                    name=run_name,
                    reinit=True,
                    config=exp_config,
                )
        else:
            wandb_run = wandb.init(
                project=f"MISTRAL_{dataset_type}",
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
        model_name,
        pretrained_model_to_load,
    )

    print("Loading a dataset...")
    print("============")
    print("SEED: ", seed)
    print("============")
    if do_pretraining:
        dataset_type = OPENWEBTEXT
        dataset = get_dataset(dataset_type, tokenizer, MISTRAL_7B)
        train_dataset, val_dataset, test_dataset = (
            dataset.get_tokenized_dataset()
        )
        sample_dataset = test_dataset
        max_seq_length = 128
        data_collator = dataset.get_data_collator()
        trainer_func = SparseTrainer
    else:
        formatting_func = get_formatting_func(dataset_type)
        max_seq_length = 1024
        train_dataset, val_dataset, test_dataset, sample_dataset = (
            get_dataset_for_instruction_tuning(dataset_type, seed,
                                               tokenizer,
                                               formatting_func,
                                               max_seq_length,
                                               train_batch_size,
                                               test_batch_size,
                                               return_sample_dataset=exp_config.set_sparsity_aware_threshold)
        )

        response_template = "Response:"
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template, tokenizer=tokenizer
        )
        trainer_func = SparseSFTTTrainer
        compute_metrics = classification_compute_metrics
        preprocess_logits_for_metrics = (
            classification_preprocess_logits_for_metrics
        )

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
        eval_steps=eval_steps,
        # early stopping counts only when eval step and save step match
        save_steps=50,
        logging_steps=5,
        save_strategy="steps",
        learning_rate=learning_rate,
        weight_decay=0.01,
        num_train_epochs=num_epochs,
        logging_dir="exp_logs",
        max_steps=max_steps,
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

    early_stopping = EarlyStoppingCallback(early_stopping_patience=20)
    sparsification_scheduler = GradualSparsificationScheduler(
        num_warmup_steps=num_warmup_steps,
        increment_ratio=increment_ratio,
        is_enabled=use_gradual_sparsification,
        model_name=model_name,
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

    print("Do pretraining: ", do_pretraining)
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
            act_hist_path = f"histograms/debugging/{exp_config.targeted_sparsity}/act_hist.pt"
        else:
            act_hist_path = f"/matx/u/lukeai/histograms/{exp_config.targeted_sparsity}/act_hist.pt"
        print("histogram: ", act_hist_path)

        activate_stats(model.get_base_model())
        if os.path.exists(act_hist_path):  # @luke TODO: remove False
            load_act_hist(model.get_base_model(), act_hist_path)
        else:
            # is_deepspeed_enabled = trainer.is_deepspeed_enabled
            # trainer.is_deepspeed_enabled = False  # A way to go around the value error when using ds stage 2 for evaluation
            # trainer.evaluate(sample_dataset)
            # trainer_config2 = copy.deepcopy(trainer_config)
            # trainer_config2.max_steps = 1
            # trainer_args2 = copy.deepcopy(trainer_args)
            # trainer_args2["args"] = trainer_config2
            # trainer2 = trainer_func(**trainer_args2)
            # trainer2.train()
            # trainer2.evaluate(test_dataset)
            trainer.evaluate(test_dataset)
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
    trainer.model = model

    if exp_config.print_sparsity:
        activate_stats(model, exp_config.print_act_stats)

    if do_evaluate:
        dataset_types = exp_config.dataset_types.split(",")
        evaluate_zero_shot(
            trainer, dataset_types, tokenizer,
            train_batch_size, test_batch_size, max_seq_length,
        )
    else:
        eval_result = trainer.evaluate(test_dataset)
    print("===Test Result==")
    print(eval_result)
    log_dict = {"test_loss": eval_result["eval_loss"]}
    if "eval_accuracy" in eval_result.keys():
        log_dict["test_accuracy"] = eval_result["eval_accuracy"]

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

    if push_to_hub and is_mainprocess():
        print("Pushing to hub")
        print("model: ", model)
        torch.save(model.state_dict(), f"/matx/u/lukeai/{model_name}.pt")
        trainer.push_to_hub()
        print(f"Pushed to hub: thrunlab/{model_name}")

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
        model_name: str = None,
        pretrained_model_to_load: str = None,
):
    print("Loading a model...")
    if is_debugging:
        config = MistralConfig(
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
        )
        model = MistralForCausalLM(config=config)
    else:
        config = MistralConfig()
        if pretrained_model_to_load == "mistralai/Mistral-7B-v0.1":
            sparse_config = get_sparse_mistral_config(
                config, use_sparse_model=False, use_sparse_regularization=False
            )
            # model = SparseMistral.from_pretrained(
            #     pretrained_model_to_load,
            #     config=sparse_config,
            #     use_flash_attention_2=True,
            # )
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_to_load,
                use_flash_attention_2=True,
            )
        else:
            sparse_config = get_sparse_mistral_config(config)
            model = SparseMistral(sparse_config)
            original_state_dict = torch.load(
                f"/matx/u/lukeai/auto.pt",
                map_location="cpu",
            )
            model.load_state_dict(original_state_dict, strict=False)
            if is_mainprocess():
                print("model1: ", model)

            # print(f"Loading from /matx/u/lukeai/{pretrained_model_to_load}.pt")
            # print(
            #     os.path.exists(f"/matx/u/lukeai/{pretrained_model_to_load}.pt")
            # )
            # state_dict = torch.load(
            #     f"/matx/u/lukeai/{pretrained_model_to_load}.pt",
            #     map_location="cpu",
            # )
            # model.load_state_dict(state_dict)

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

    if is_mainprocess():
        print("model2: ", model)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Applying LoRA...")
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=128,
        bias="none",
        target_modules=[
            "q_proj",
            "v_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()

    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for text classification"
    )
    parser.add_argument("--dataset_types", type=str,
                        help="List of dataset types")
    parser.add_argument(
        "--do_evaluate",
        action="store_true",
        default=False,
        help="Whether to evaluate the model without training",
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
        "--pretrained_model_to_load",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Name of the model",
    )
    parser.add_argument(
        "--dataset_type", type=str, default="cola", help="Dataset Type"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ckpt",
        help="Output Directory",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=8, help="Number of training epochs"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=50, help="Evaluation steps"
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
