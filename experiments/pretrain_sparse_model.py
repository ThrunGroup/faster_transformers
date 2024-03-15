from experiments.models.sparse_mistral.sparsification_sftt import (
    SparseTrainer,
    GracefulRegularizationScheduler,
)
from utils.constants import (
    REFINED_WEB,
)
from utils.utils import ds_print, is_mainprocess, is_running_deepspeed
from experiments.data.get_dataset import get_dataset
from mistral_instruct_tuning import *


def get_run_name(exp_config):
    model_name = exp_config.model_name
    dataset_type = exp_config.dataset_type

    run_name = f"{model_name}_{dataset_type}"

    if exp_config.use_relu:
        run_name += "_relu"

    if exp_config.use_graceful_regularization:
        run_name += "_graceful_reg"

    if exp_config.targeted_sparsity:
        run_name += f"_{int(exp_config.targeted_sparsity * 100)}p"

    return run_name


def train(exp_config, use_wandb: bool = True, use_sweep: bool = False):
    ds_print(f"Config: {exp_config}")
    if use_wandb and is_mainprocess():
        wandb.init()
        if use_sweep:
            exp_config = wandb.config
        else:
            wandb.config = exp_config
        if is_running_deepspeed():
            time.sleep(1)

    model_name = exp_config.model_name
    gradient_checkpointing = exp_config.gradient_checkpointing
    push_to_hub = exp_config.push_to_hub
    is_debugging = exp_config.is_debugging
    dataset_type = exp_config.dataset_type
    is_plot = exp_config.is_plot
    is_first_training = exp_config.is_first_training

    # Graceful Regularization
    num_warmup_steps = exp_config.num_warmup_steps
    keep_regularization_with_kill = exp_config.keep_regularization_with_kill
    use_graceful_regularization = exp_config.use_graceful_regularization

    use_sparse_regularization = exp_config.use_sparse_regularization
    if use_graceful_regularization and not is_first_training:
        print("No sparse reg!")
        use_sparse_regularization = False

    print("FIRST TRAINING? : ", is_first_training)

    if is_debugging:
        act_hist_path = f"/scr/lukeai/act_hist/debugging_{model_name}/{dataset_type}.pt"
    else:
        act_hist_path = f"/scr/lukeai/act_hist/{model_name}/{dataset_type}.pt"

    # If not using sparse Mistral model, all flags related to sparse model should be set as zero or false.
    if not exp_config.use_sparse_model:
        exp_config.use_sparse_regularization = (
            exp_config.print_sparsity
        ) = exp_config.print_act_stats = exp_config.set_sparsity_aware_threshold = False
        exp_config.targeted_sparsity = 0

    if exp_config.use_relu:
        exp_config.targeted_sparsity = "ReLU"

    run_name = get_run_name(exp_config)

    # Load model
    model, tokenizer, config = prepare_sparse_model(
        is_debugging,
        exp_config.use_sparse_model,
        use_sparse_regularization,
        exp_config.use_spm,
        exp_config.use_graceful_regularization,
        exp_config.sparse_model_dir,
        exp_config.use_lora,
        exp_config.use_flash_attn,
        base_model_name=MISTRAL_7B,
        gradient_checkpointing=gradient_checkpointing,
        use_relu=exp_config.use_relu,
    )

    # Load dataset
    dataset = get_dataset(dataset_type, tokenizer, MISTRAL_7B, max_seq_length=exp_config.max_seq_length)
    train_dataset, val_dataset, test_dataset = dataset.get_tokenized_dataset()
    print(train_dataset, val_dataset, test_dataset)
    data_collator = dataset.get_data_collator()

    if is_debugging:
        output_dir = "debugging_output"
    else:
        output_dir = os.path.join(exp_config.output_dir, run_name)
    with open("model_directory.txt", "w") as f:
        f.write(output_dir)

    # Use only 500 samples of training dataset to collect statistics
    sampled_train_dataset = train_dataset.select(range(500))

    if use_wandb:
        if is_debugging:
            run_name += "_debugging"
        run_name += "_" + get_datetime(True)
        wandb.name = run_name

        if is_mainprocess():
            wandb.init(
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
        num_train_epochs=exp_config.num_epochs,
        logging_dir="exp_logs",
        save_total_limit=1,
        greater_is_better=False,
        gradient_accumulation_steps=exp_config.gradient_accumulation_steps,
        deepspeed=exp_config.ds_config_path if not is_debugging else None,
        per_device_train_batch_size=exp_config.train_batch_size,
        per_device_eval_batch_size=exp_config.test_batch_size,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True if not is_debugging else False,
        hub_model_id=f"thrunlab/{run_name}",
        push_to_hub=push_to_hub,
        seed=exp_config.seed,
        data_seed=exp_config.seed,
    )

    graceful_regularization_scheduler = GracefulRegularizationScheduler(
        num_warmup_steps=num_warmup_steps,
        is_enabled=use_graceful_regularization and is_first_training,
        model_name=model_name,
        test_dataset=test_dataset,
        targeted_sparsity=exp_config.targeted_sparsity,
        keep_regularization_with_kill=keep_regularization_with_kill,
    )

    # See https://discuss.huggingface.co/t/using-iterabledataset-with-trainer-iterabledataset-has-no-len/15790/2
    trainer = SparseTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=trainer_config,
        train_dataset=train_dataset.with_format("torch"),  # train_dataset.with_format("torch"),
        eval_dataset=val_dataset.with_format("torch"),
        # compute_metrics=dataset.compute_metrics,
        # preprocess_logits_for_metrics=dataset.preprocess_logits_for_metrics,
        callbacks=[graceful_regularization_scheduler],
        use_sparse_regularization=use_sparse_regularization,
    )
    graceful_regularization_scheduler.set_trainer(trainer)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        config.use_cache = False
        model.config.use_cache = False
        # model.enable_input_require_grads()  # See https://github.com/huggingface/transformers/issues/23170

    if exp_config.use_lora:
        base_model = model.get_base_model()
    else:
        base_model = model

    # Collect statistics and find the threshold for given sparsity
    if not use_graceful_regularization and exp_config.set_sparsity_aware_threshold:
        activate_stats(base_model)
        if os.path.exists(act_hist_path):
            load_act_hist(base_model, act_hist_path)
        elif not exp_config.use_relu:
            is_deepspeed_enabled = trainer.is_deepspeed_enabled
            trainer.is_deepspeed_enabled = False  # A way to go around the value error when using ds stage 2 for evaluation
            trainer.evaluate(sampled_train_dataset)
            trainer.is_deepspeed_enabled = is_deepspeed_enabled
            save_act_hist(base_model, act_hist_path)

        if exp_config.use_relu:
            set_sparse_threshold(base_model, 0, True)
        else:
            set_sparse_threshold(base_model, exp_config.targeted_sparsity)

    if not use_graceful_regularization and exp_config.print_sparsity:
        if exp_config.use_lora:
            base_model = model.get_base_model()
        else:
            base_model = model
        activate_stats(base_model, exp_config.print_act_stats)

        is_deepspeed_enabled = trainer.is_deepspeed_enabled
        trainer.is_deepspeed_enabled = False
        eval_result = trainer.evaluate(sampled_train_dataset)
        trainer.is_deepspeed_enabled = is_deepspeed_enabled

        print("===Pre-training test Result==")
        print(eval_result)
        total_sparsity = print_dead_neuron_stats(base_model)
        print(f"pre-training sparsity: {total_sparsity}")

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
            warnings.warn("use_spm arg is ignored as use_spare_model arg is not activated.")

    trainer.train(resume_from_checkpoint=not is_first_training)

    # Evaluate on test dataset
    if exp_config.use_lora:
        model = model.merge_and_unload()

    # Enable sparse predictor in inference stage
    if exp_config.use_spm:
        if exp_config.use_sparse_model:
            enable_sparse_predictor(model)
        else:
            warnings.warn("use_spm arg is ignored as use_spare_model arg is not activated.")

    if exp_config.print_sparsity:
        activate_stats(model, exp_config.print_act_stats)
    eval_result = trainer.evaluate(sampled_train_dataset)
    print("===Test Result==")
    print(eval_result)
    log_dict = {}

    if exp_config.print_sparsity:
        total_sparsity = print_dead_neuron_stats(model)
        log_dict.update({"total_sparsity": total_sparsity})

    if exp_config.print_act_stats:
        save_act_hist(model, act_hist_path)
        plot_act(model)

    if exp_config.model_save:
        # Save thresholds
        if exp_config.use_sparse_model:
            thresholds = [float(m.mlp.dead_threshold) for m in model.model.layers]
            model.config.thresholds = thresholds
        # model.config.save_pretrained(output_dir)
        model.save_pretrained(output_dir + "_no_adapter")
        tokenizer.save_pretrained(output_dir + "_no_adapter")

    if push_to_hub:
        trainer.model = model
        trainer.deepspeed = model
        trainer.push_to_hub()

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
    args = parse_args()
    args.dataset_type = REFINED_WEB
    train(args, use_wandb=args.use_wandb)
