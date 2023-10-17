import torch
import numpy as np
from torch import nn
from transformers.models.t5.modeling_t5 import (
    T5LayerSelfAttention,
    T5Attention,
)

from experiments.models.t5_mare_upcycling.mare_mlp import MAREMLP
from experiments.models.t5_mare_upcycling.mare_attention import (
    MAREAttention,
    PruningAttention,
    PruningSelfAttention,
)


def inject_pretrained_weights_to_mlp(
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


def inject_pretrained_weights_to_pruning_attention(
    new_attention,
    original_attention,
):
    new_attention.SelfAttention = original_attention.SelfAttention
    new_attention.layer_norm = original_attention.layer_norm


def apply_pruning_self_attention_t5(
    model,
    config,
):
    print("Applying Pruning Attention to T5...")

    # Iterate through blocks and replace self-attention layers
    # for block in [model]
    for is_decoder, network in enumerate(
        [model.transformer.encoder, model.transformer.decoder]
    ):
        for block_idx, block in enumerate(network.block):
            original_attention_layer = block.layer[0].SelfAttention
            has_relative_attention_bias = (
                original_attention_layer.has_relative_attention_bias
            )

            new_attention_layer = T5LayerSelfAttention(
                config,
                has_relative_attention_bias,
            )

            new_attention_layer.SelfAttention = PruningSelfAttention(
                config,
                has_relative_attention_bias,
            )
            new_attention_layer.SelfAttention.q = original_attention_layer.q
            new_attention_layer.SelfAttention.k = original_attention_layer.k
            new_attention_layer.SelfAttention.v = original_attention_layer.v
            new_attention_layer.SelfAttention.o = original_attention_layer.o

            block.layer[0] = new_attention_layer

    # freeze the model
    for name, param in model.named_parameters():
        if not (
            "pruning_scalers" in name
            or "classification" in name
            or "SelfAttention" in name
        ):
            print(name)
            param.requires_grad = False


def apply_pruning_attention_t5(
    model,
    config,
):
    print("Applying Pruning Attention to T5...")

    # Iterate through blocks and replace self-attention layers
    # for block in [model]
    for is_decoder, network in enumerate(
        [model.transformer.encoder, model.transformer.decoder]
    ):
        for block_idx, block in enumerate(network.block):
            original_attention_layer = block.layer[0]
            has_relative_attention_bias = (
                original_attention_layer.SelfAttention.has_relative_attention_bias
            )

            new_attention_layer = PruningAttention(
                config,
                has_relative_attention_bias,
            )

            inject_pretrained_weights_to_pruning_attention(
                new_attention_layer,
                original_attention_layer,
            )
            block.layer[0] = new_attention_layer

    # freeze the model
    for name, param in model.named_parameters():
        if not (
            "pruning_scalers" in name
            or "classification" in name
            or "SelfAttention" in name
        ):
            print(name)
            param.requires_grad = False


def apply_mare_attention_t5(
    model,
    config,
    num_generalists: int = 0,
    num_experts: int = 0,
    capacity_factor: int = 0,
    apply_to_half_layers: bool = False,
):
    print("Applying MARE Attention to T5...")
    print(
        "Original num params: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # Iterate through blocks and replace self-attention layers
    # for block in [model]
    for is_decoder, network in enumerate(
        [model.transformer.encoder, model.transformer.decoder]
    ):
        for block_idx, block in enumerate(network.block):
            original_attention_layer = block.layer[0].SelfAttention
            has_relative_attention_bias = (
                original_attention_layer.has_relative_attention_bias
            )
            print(has_relative_attention_bias)

            new_attention_layer = MAREAttention(
                config,
                has_relative_attention_bias,
                num_all_rounders=num_generalists,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
            )

            inject_pretrained_weights_to_attention(
                new_attention_layer,
                original_attention_layer,
                num_all_rounders=num_generalists,
                num_experts=num_experts,
            )
            block.layer[0] = new_attention_layer

    print(
        "Changed num params:  ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )


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

            inject_pretrained_weights_to_mlp(
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
