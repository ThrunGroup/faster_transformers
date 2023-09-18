from experiments.models.gpt2_peft_mare.model import MAREMLP


def inject_pretrained_weights(new_mlp, original_mlp):
    new_mlp.c_fc.weight = original_mlp.mlp.c_fc.weight
    new_mlp.c_fc.bias = original_mlp.mlp.c_fc.bias
    new_mlp.c_proj.weight = original_mlp.mlp.c_proj.weight
    new_mlp.c_proj.bias = original_mlp.mlp.c_proj.bias


def apply_mare_lora(
    model,
    config,
    num_all_rounders: int = 0,
    num_experts: int = 0,
):
    print("Applying MARE LoRA...")
    print(f"All Rounders: {num_all_rounders} / Experts: {num_experts}")
    for block in model.transformer.h:
        intermediate_size = block.mlp.c_fc.weight.shape[1]
        new_mlp = MAREMLP(
            intermediate_size, config, num_all_rounders, num_experts
        )
        inject_pretrained_weights(
            new_mlp,
            block,
        )
        block.mlp = new_mlp


def freeze_model(model):
    for name, param in model.named_parameters():
        if "mlp" not in name:
            param.requires_grad = False

    print(
        "Total params: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    print("Trainable params: ", sum(p.numel() for p in model.parameters()))
    print(
        "Percentage: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        / sum(p.numel() for p in model.parameters())
        * 100,
    )
