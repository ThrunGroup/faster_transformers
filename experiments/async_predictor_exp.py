from mistral_instruct_tuning import train, parse_args


def main(targeted_sparsity):
    args = parse_args()
    args.use_sparse_model = True
    args.is_debugging = False
    args.model_save = True
    args.targeted_sparsity = targeted_sparsity
    args.set_sparsity_aware_threshold = True
    args.print_sparsity = True
    args.use_lora = True
    args.num_epochs = 5
    # args.gradient_checkpointing = True
    # train(args)

    args.use_spm = True
    args.use_lora = False
    args.model_name = f"Mistral_Sparse_Async_{targeted_sparsity}"
    args.print_sparsity = False
    args.set_sparsity_aware_threshold = False
    args.sparse_model_dir = f"/scr/jay/ckpt/Mistral_Sparse_cola_{targeted_sparsity}"
    train(args)


if __name__ == "__main__":
    for sparsity in [0.9, 0.85, 0.75, 0.65, 0.95]:
        main(sparsity)
