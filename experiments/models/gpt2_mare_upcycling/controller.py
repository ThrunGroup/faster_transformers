from .model import MAREMLP


def inject_pretrained_weights(
    new_mlp,
    original_block,
    config,
    num_all_rounders: int = 0,
    num_experts: int = 0,
):
    # Multiply the first layer
    c_fc_weight = original_block.mlp.c_fc.weight.data  # (in, out)
    c_fc_bias = original_block.mlp.c_fc.bias.data

    # Multiply the second layer
    c_proj_weight = original_block.mlp.c_proj.weight.data  # (in, out)
    c_proj_bias = original_block.mlp.c_proj.bias.data

    # Inject the pretrained weights
    if num_all_rounders:
        new_mlp.c_fc_ar.weight.data = c_fc_weight.repeat(num_all_rounders, 1)
        new_mlp.c_fc_ar.bias.data = c_fc_bias.repeat(1)
        new_mlp.c_proj_ar.weight.data = c_proj_weight.repeat(1, num_all_rounders)
        new_mlp.c_proj_ar.bias.data = c_proj_bias.repeat(num_all_rounders)

    if num_experts:
        new_mlp.c_fc_ex.weight.data = c_fc_weight.repeat(num_experts, 1)
        new_mlp.c_fc_ex.bias.data = c_fc_bias.repeat(1)
        new_mlp.c_proj_ex.weight.data = c_proj_weight.repeat(1, num_experts)
        new_mlp.c_proj_ex.bias.data = c_proj_bias.repeat(num_experts)


def apply_mare_upcycling(
    model,
    config,
    num_generalists: int = 0,
    num_experts: int = 0,
    capacity_factor: int = 0,
):
    print("Applying MARE Upcycling...")
    print(sum(p.numel() for name, p in model.named_parameters() if p.requires_grad))

    # Iterate through blocks and replace self-attention layers
    for block in model.transformer.h:
        intermediate_size = block.mlp.c_fc.weight.shape[1]
        new_attention = MAREMLP(
            intermediate_size,
            config,
            num_all_rounders=num_generalists,
            num_experts=num_experts,
            capacity_factor= capacity_factor,
        )
        inject_pretrained_weights(
            new_attention,
            block,
            config,
            num_all_rounders=num_generalists,
            num_experts=num_experts,
        )
        block.mlp = new_attention

    print(sum(p.numel() for name, p in model.named_parameters() if p.requires_grad))
