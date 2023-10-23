from torch import nn
from transformers.pytorch_utils import Conv1D

from .model import MAREAttention


def inject_pretrained_weights(
    new_attention, original_block, config, num_generalists
):
    num_heads = config.n_head
    embed_dim = config.n_embd

    # original_attn = original_block.attn.c_attn
    new_attention.expert_attn = original_block.attn.c_attn
    new_attention.generalist_attn = original_block.attn.c_attn

    # Inject the pretrained output projection weights into the new MARE model
    new_attention.c_proj = original_block.attn.c_proj
    new_attention.generalist_proj = original_block.attn.c_proj


def apply_mare_attention(
    model, config, random_token_selection, num_generalists: int = 4
):
    print("Applying MARE attention...")
    # Iterate through blocks and replace self-attention layers
    for i, block in enumerate(model.transformer.h):
        new_attention = MAREAttention(
            config,
            layer_idx=i,
            random_token_selection=random_token_selection,
            num_generalists=num_generalists,
        )
        inject_pretrained_weights(
            new_attention, block, config, num_generalists
        )
        block.attn = new_attention
