from instruction_tuning import train, parse_args
from transformers import MistralConfig
from experiments.models.sparse_mistral.sparsification_sftt import (
    SparseSFTTTrainer,
    SparseMistralforCausalLM,
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


def main(targeted_sparsity):
    args = parse_args()
    args.use_sparse_model = True
    args.model_save = True
    args.targeted_sparsity = targeted_sparsity
    args.set_sparsity_aware_threshold = True
    args.print_sparsity = True
    args.use_lora = True
    args.num_epochs = 5
    args.use_spm = False
    # args.gradient_checkpointing = True
    print(args)
    # train(args)

    args.use_spm = True
    args.use_lora = False
    args.model_name = f"Mistral_Sparse_Async"
    args.print_sparsity = False
    args.set_sparsity_aware_threshold = False
    args.sparse_model_dir = f"/scr/jay/ckpt/Mistral_Sparse_cola_{targeted_sparsity}"
    print(args)
    train(args)


if __name__ == "__main__":
    for sparsity in [0.9, 0.85, 0.75, 0.65, 0.95]:
        main(sparsity)
