import torch
import numpy as np
from torch import nn
from transformers.models.t5.modeling_t5 import (
    T5LayerSelfAttention,
    T5Attention,
)

from experiments.models.t5_mare_downcycling.model import (
    MAREDownCyclingSelfAttention,
    MAREDownCyclingAttention,
)


def apply_mare_downcycling_attention(
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
            original_self_attention = original_attention_layer.SelfAttention
            has_relative_attention_bias = (
                original_attention_layer.SelfAttention.has_relative_attention_bias
            )

            new_attention_layer = MAREDownCyclingSelfAttention(
                config,
                has_relative_attention_bias,
            )

            new_attention_layer.SelfAttention = MAREDownCyclingAttention(
                config,
                has_relative_attention_bias,
                num_all_rounders=1,
            )
            new_attention_layer.SelfAttention.q = original_self_attention.q
            new_attention_layer.SelfAttention.k = original_self_attention.k
            new_attention_layer.SelfAttention.v = original_self_attention.v
            new_attention_layer.SelfAttention.o = original_self_attention.o
            new_attention_layer.layer_norm = (
                original_attention_layer.layer_norm
            )

            block.layer[0] = new_attention_layer
