import torch
import numpy as np
from torch import nn

from experiments.models.t5_mare_upcycling.model import MAREMLP


def inject_pretrained_weights(
    new_mlp,
    original_mlp,
    num_all_rounders: int = 0,
    num_experts: int = 0,
):
    # Multiply the first layer
    wi = original_mlp.DenseReluDense.wi.weight.data

    # Multiply the second layer
    wo = original_mlp.DenseReluDense.wo.weight.data

    # Inject the pretrained weights
    if num_all_rounders:
        new_mlp.wi_ar.weight.data = wi.repeat(1, num_all_rounders)
        new_mlp.wo_ar.weight.data = wo.repeat(num_all_rounders, 1)

        if num_all_rounders >= 2:
            with torch.no_grad():
                new_mlp.wi_ar.weight.data += torch.randn_like(
                    new_mlp.wi_ar.weight
                ) / np.sqrt(new_mlp.wi_ar.weight.shape[1])
                new_mlp.wo_ar.weight.data += (
                    torch.randn_like(new_mlp.wo_ar.weight)
                    * 2
                    / np.sqrt(new_mlp.wo_ar.weight.shape[1])
                )

    if num_experts:
        new_mlp.wi_ex.weight.data = wi.repeat(1, num_experts)
        new_shape = new_mlp.wi_ex.weight.shape
        new_mlp.wi_ex.weight.data += torch.randn(new_shape) / np.sqrt(
            new_shape[1]
        )

        new_mlp.wo_ex.weight.data = wo.repeat(num_experts, 1)
        new_shape = new_mlp.wo_ex.weight.shape
        new_mlp.wo_ex.weight.data += torch.randn(new_shape) / np.sqrt(
            new_shape[1]
        )

    del new_mlp.wi
    del new_mlp.wo


def apply_mare_upcycling_t5(
    model,
    config,
    num_generalists: int = 0,
    num_experts: int = 0,
    capacity_factor: int = 0,
    apply_to_half_layers: bool = False,
):
    print("Applying MARE Upcycling to T5...")
    print(
        "Original num params: ",
        sum(
            p.numel()
            for name, p in model.named_parameters()
            if p.requires_grad
        ),
    )

    # Iterate through blocks and replace self-attention layers
    # for block in [model]
    for is_decoder, network in enumerate(
        [model.transformer.encoder, model.transformer.decoder]
    ):
        for block_idx, block in enumerate(network.block):
            if apply_to_half_layers and block_idx % 2 == 1:
                # apply MoE to half the layers
                continue

            if is_decoder:
                mlp_layer = block.layer[2]
            else:
                mlp_layer = block.layer[1]

            new_mlp = MAREMLP(
                config,
                num_all_rounders=num_generalists,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
            )

            inject_pretrained_weights(
                new_mlp,
                mlp_layer,
                num_all_rounders=num_generalists,
                num_experts=num_experts,
            )
            mlp_layer.DenseReluDense = new_mlp

    print(
        "Changed num params:  ",
        sum(
            p.numel()
            for name, p in model.named_parameters()
            if p.requires_grad
        ),
    )
