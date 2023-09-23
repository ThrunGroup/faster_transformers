import os
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoTokenizer,
)
import wandb
import evaluate

from data.get_dataset import get_dataset
from evaluate_models import evaluate_model
from models.build_model import build_model
from utils.constants import (
    # Models
    GPT2,
    T5,
    # Datasets
    # SST2,
    SAMSUM,
    BILLSUM,
    SQUAD,
    QNLI,
    SST2,
    COLA,
)


def finetune(
    model_type: str,
    model_name: str,
    model,
    tokenizer,
    dataset_name: str,
    num_epochs: int = 10,
    resume_from_checkpoint: bool = False,
    random_token_selection: bool = False,
    run_name: str = "",
    learning_rate: float = 0.001,
    do_evaluation: bool = True,
):
    """
    Finetune the model on a dataset

    :param model_name: Name of the model to finetune
    :param model: PyTorch model to finetune
    :param tokenizer: Tokenizer to tokenize the dataset
    :param dataset_name: Name of the dataset to train on
    :param num_epochs: Number of epochs to train the model
    :param resume_from_checkpoint: Whether to resume the training from the checkpoint
    :param do_evaluation: Whether to evaluate the model after training and save the metrics
    :returns: Dictionary of evaluation metrics and inference time if `do_evaluation` is true
    """
    dataset = get_dataset(dataset_name, model_type, tokenizer, model)
    train_dataset, test_dataset = dataset.get_tokenized_dataset()
    data_collator = dataset.get_data_collator()
    # compute_metrics = dataset.get_compute_metrics()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "checkpoints", model_name)

    if "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) == 0:
        wandb_run = wandb.init(
            project="gpt2_billsum",
            name=run_name,
            reinit=True,
        )

    metric = evaluate.load("f1")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        print(preds.shape, labels.shape)
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    def preprocess_logits_for_metrics(logits, labels):
        # print(len(logits), logits[0].shape, logits[1].shape, labels.shape)
        return logits[0].argmax(dim=-1)

    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=1,
        save_strategy="steps",
        learning_rate=5e-5,
        warmup_steps=500,
        per_device_train_batch_size=64,  # 32 for SMALL, 12 for MED
        per_device_eval_batch_size=64,  # 32 for MED, XL, 64 for SMALL
        weight_decay=0.01,
        num_train_epochs=num_epochs,
        logging_dir="exp_logs",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=run_name,
        # for memory issues
        gradient_accumulation_steps=8,
        deepspeed="ds_config.json",
        gradient_checkpointing=False,
        fp16=True,
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=10, early_stopping_threshold=0.001
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        callbacks=[early_stopping],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )

    print("\n*********************")
    print("Training ", model_name)
    print("*********************\n")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if do_evaluation:
        return evaluate_model(
            model_name,
            dataset_name,
            model=model,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            trainer=trainer,
        )

    if "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) == 0:
        wandb_run.finish()


def create_model(
    model_type: str,
    dataset_name: str,
    learning_rate: float,
    num_epochs: int = 5,
    random_token_selection: bool = False,
    run_name: str = "",
    capacity_factor: int = 0,
    use_mare_lora: bool = False,
    use_mare_attention: bool = False,
    use_lora: bool = False,
    apply_to_half_layers: bool = False,
    do_evaluation: bool = True,
    **kwargs,
):
    """
    Load a naive pretrained model from HuggingFace, apply the accelerator to FC layers,
    and finetune it on the specified dataset.

    :param model_type: Type of the model (TXL, GPT, etc.)
    :param dataset_name: Name of the dataset to train on
    :param num_epochs: Number of epochs to finetune the model
    :param do_evaluation: Whether to evaluate the model after training and save the metrics
    :return: Dictionary of evaluation logs if `do_evaluation` is true
    """
    # Build a Gated Attention model
    model_name = model_type
    if use_lora:
        model_name += "_lora"
    if use_mare_attention:
        model_name += "_gated"

    model = build_model(
        model_type,
        dataset_name,
        use_mare_lora=use_mare_lora,
        use_mare_attention=use_mare_attention,
        random_token_selection=random_token_selection,
        capacity_factor=capacity_factor,
        apply_to_half_layers=apply_to_half_layers,
        **kwargs,
    )
    model.config.pad_token_id = model.config.eos_token_id

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.pad_token = tokenizer.eos_token

    finetune(
        model_type,
        model_name,
        model,
        tokenizer,
        dataset_name,
        num_epochs,
        resume_from_checkpoint=False,
        learning_rate=learning_rate,
        run_name=run_name,
        random_token_selection=random_token_selection,
    )


if __name__ == "__main__":
    common_args = {
        "model_type": T5,
        "dataset_name": COLA,
        "num_epochs": 5,
        "learning_rate": 5e-5,
    }

    # create_model(
    #     run_name="baseline",
    #     **common_args,
    # )

    # create_model(
    #     use_mare_upcycling=True,
    #     num_generalists=2,
    #     num_experts=8,
    #     capacity_factor=4,
    #     run_name="c4_mare_ar1_ex12_1/4",
    #     **common_args,
    # )

    create_model(
        use_mare_upcycling=True,
        capacity_factor=2,
        num_generalists=1,
        num_experts=8,
        run_name="cola_c2_mare_ar1_ex8",
        **common_args,
    )
    create_model(
        use_mare_upcycling=True,
        capacity_factor=2,
        num_generalists=2,
        num_experts=0,
        run_name="cola_c2_mare_ar2",
        **common_args,
    )

    create_model(
        use_mare_upcycling=True,
        num_generalists=0,
        capacity_factor=2,
        num_experts=16,
        run_name="cola_c2_mare_ex16",
        **common_args,
    )

    # do
    # create_model(
    #     use_mare_upcycling=True,
    #     num_generalists=1,
    #     capacity_factor=2,
    #     num_experts=2,
    #     run_name="c2_mare_1ar_ex16_1/32",
    #     **common_args,
    # )

    # needs some updates. training is very unstable.
    # create_model(
    #     use_mare_upcycling=True,
    #     capacity_factor=2,
    #     num_generalists=2,
    #     num_experts=0,
    #     run_name="c2_moe_soft_ar2",
    #     **common_args,
    # )

    # create_model(
    #     use_mare_upcycling=True,
    #     num_generalists=2,
    #     num_experts=0,
    #     run_name="mare_ar2_ex0",
    #     **common_args,
    # )
