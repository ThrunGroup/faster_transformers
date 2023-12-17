import copy
import accelerate

from utils.constants import MISTRAL_7B, BOOLQ, MULTIRC
from create_models import create_model
from transformers import (
    AutoModelForSequenceClassification,
    MistralForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
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


def _get_trainable_parameter_groups(trainer):
    param_groups = []
    for param_group in trainer.optimizer.param_groups:
        trainable_params = {
            "params": [p for p in param_group["params"] if p.requires_grad]
        }
        param_groups.append(trainable_params)
    return param_groups

def train_mistral(
    dataset_type: str,
    use_relu: bool = True,
    use_quantization: bool = False,
    use_lora: bool = True,
    pretrained_dir: str = MISTRAL_7B,

):
    # dataset_type = BOOLQ  # @jay: fix hard-coded
    model_name = "mistral7b"

    os.environ["WANDB_PROJECT"] = f"Mistral 7b on {dataset_type}"
    os.environ["WANDB_WATCH"] = "gradients"
    home_dir = "/scr/jay"
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    if use_quantization:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MISTRAL_7B,
            # device_map="auto",
            quantization_config=nf4_config,
            use_cache=False,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_dir,
            # device_map="auto",
            use_cache=False,
            torch_dtype=torch.bfloat16,
        )

    if use_relu:
        model.config.hidden_act = "relu"
    model.config.pad_token_id = (
        model.config.eos_token_id
    )  # pad token id is required when calculating loss in the
    # forward pass of sequence classification model

    tokenizer = AutoTokenizer.from_pretrained(MISTRAL_7B)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    output_dir = f"mistral7b_{dataset_type}"

    args_classifier = TrainingArguments(
        output_dir=os.path.join(home_dir, output_dir) + "_classifier",
        num_train_epochs=1,
        max_steps=5,
        # per_device_train_batch_size=4,
        # max_steps=1000,  # comment out this line if you want to train in epochs
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=10,
        logging_steps=10,
        save_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=50,  # comment out this line if you want to evaluate at the end of each epoch
        learning_rate=2e-4,
        gradient_checkpointing=True,
        # weight_decay=0.01,
        bf16=True,
        # fp16=True,
        lr_scheduler_type="constant",
        hub_model_id=f"thrunlab/{output_dir}_classifier",
        push_to_hub=True,
        seed=1,
        data_seed=1,
        save_steps=600,
        report_to="wandb",
        run_name=output_dir,
        load_best_model_at_end=True,
        deepspeed="mistral_ds_config.json",
    )

    args_finetune = copy.deepcopy(args_classifier)
    args_finetune.output_dir = os.path.join(home_dir, output_dir) + "_finetuned"
    args_finetune.learning_rate = 5e-5
    args_finetune.hub_model_id = f"thrunlab/{output_dir}_finetuned"

    # Get dataset
    dataset = get_dataset(dataset_type, tokenizer, model, MISTRAL_7B)
    train_dataset, val_dataset, test_dataset = dataset.get_tokenized_dataset()
    max_seq_length = 131072

    # Call relevant functions
    data_collator = dataset.get_data_collator()
    compute_metrics = dataset.get_compute_metrics()
    # preprocess_logits_for_metrics = dataset.preprocess_logits_for_metrics  # Don't need this as
    # MistralForSequenceClassification only uses last logits (pooled_logits)
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=8, early_stopping_threshold=0.01
    )

    trainer_classifier = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args_classifier,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[early_stopping],
    )

    # Freeze base model
    for params in model.model.parameters():  # Exclude classifier head
        params.requires_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    # trainer_classifier.optimizer.param_groups = _get_trainable_parameter_groups(trainer_classifier)
    trainer_classifier.train()
    trainer_classifier.evaluate(test_dataset)

    # Unfreeze Base model for fine-tuning
    for params in model.model.parameters():
        params.requires_grad = True

    # trainer_classifier.push_to_hub()
    # trainer_classifier.save_model()
    # accelerator = trainer_classifier.accelerator
    # unwrapped_model = accelerator.unwrap_model(model)

    # New Code #
    # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
    # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
    # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
    # For Zero Stages 1 and 2, models are saved as usual in the output directory.
    # The model name saved is `pytorch_model.bin`
    # unwrapped_model.save_pretrained(
    #     args_classifier.output_dir,
    #     is_main_process=accelerator.is_main_process,
    #     save_function=accelerator.save,
    #     state_dict=accelerator.get_state_dict(model),
    # )
    # from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
    # checkpoint_dir = os.path.join(trainer_classifier.args.output_dir, "checkpoint-final")
    # trainer_classifier.deepspeed.save_checkpoint(checkpoint_dir)
    # fp32_model = load_state_dict_from_zero_checkpoint(trainer_classifier.model, checkpoint_dir)
    # fp32_model.save_pretrained(args_classifier.output_dir)
    # unwrapped_model = trainer_classifier.accelerator.unwrap_model(trainer_classifier.deepspeed)
    # unwrapped_model.save_pretrained(args_classifier.output_dir)

    # trainer_classifier.model.save_pretrained(args_classifier.output_dir)
    # clear memory
    del model
    del trainer_classifier
    torch.cuda.empty_cache()

    model = AutoModelForSequenceClassification.from_pretrained(
            args_classifier.output_dir,
            # device_map="auto",
            use_cache=False,
            torch_dtype=torch.bfloat16,
        )

    if use_relu:
        model.config.hidden_act = "relu"
    model.config.pad_token_id = model.config.eos_token_id

    # Unfreeze Base model for fine-tuning
    # for params in model.model.parameters():
    #     params.requires_grad = True

    if use_quantization:
        model = prepare_model_for_kbit_training(model)

    if use_lora:
        peft_config = LoraConfig(
            lora_alpha=16,  # Scaling factor for the weight matrices
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, peft_config)  # PeftModel.base_model --> LoraModel/ LoraModel.model = model

    trainer_finetune = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args_finetune,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[early_stopping],
    )

    # transformer_model is the base mistral model (excluding classifier head)
    # if use_lora:
    #     transformer_model = model.base_model.model.model
    # else:
    #     transformer_model = model.model

    # trainer_finetune.optimizer.param_groups = _get_trainable_parameter_groups(trainer_finetune)
    trainer_finetune.train()
    # trainer_finetune.push_to_hub()
    trainer_finetune.evaluate(test_dataset)

    del trainer_finetune
    del model
    torch.cuda.empty_cache()

def main():
    home_dir = "/scr/jay"


    for dataset_type in [BOOLQ, MULTIRC]:
        output_dir = f"mistral7b_{dataset_type}_finetuned"
        output_dir = os.path.join(home_dir, output_dir)
        train_mistral(dataset_type, False, False)
        train_mistral(dataset_type, True, False, output_dir)
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


if __name__ == "__main__":
    main()
