from torch import nn
from transformers.pytorch_utils import Conv1D

from .model import MAREAttention


def inject_pretrained_weights(
    new_attention, original_block, config, num_generalists
):
    num_heads = config.n_head
    embed_dim = config.n_embd
    head_dim = embed_dim // num_heads
    num_experts = num_heads - num_generalists

    original_attn = original_block.attn.c_attn

    # Extract attention weights and biases for generalists from the pretrained
    # attention layer.
    generalist_attn_weight = (
        original_attn.weight.view(embed_dim, 3, -1)[
            : num_generalists * head_dim, :, : num_generalists * head_dim
        ]
        .contiguous()
        .view(num_generalists * head_dim, -1)
    )
    generalist_attn_bias = (
        original_attn.bias.view(3, -1)[:, : num_generalists * head_dim]
        .contiguous()
        .view(-1)
    )

    # Extract attention weights and biases for experts
    expert_attn_weight = (
        original_attn.weight.view(embed_dim, 3, -1)[
            num_generalists * head_dim :, :, num_generalists * head_dim :
        ]
        .contiguous()
        .view(num_experts * head_dim, -1)
    )
    expert_attn_bias = (
        original_attn.bias.view(3, -1)[:, num_generalists * head_dim :]
        .contiguous()
        .view(-1)
    )

    # Inject the new attention weights into the new MARE model
    generalist_in_features = num_generalists * head_dim
    generalist_out_features = 3 * generalist_in_features
    generalist_attn = Conv1D(generalist_out_features, generalist_in_features)
    generalist_attn.weight = nn.Parameter(generalist_attn_weight)
    generalist_attn.bias = nn.Parameter(generalist_attn_bias)
    new_attention.generalist_attn = generalist_attn

    expert_in_features = num_experts * head_dim
    expert_out_features = 3 * expert_in_features
    expert_attn = Conv1D(expert_out_features, expert_in_features)
    expert_attn.weight = nn.Parameter(expert_attn_weight)
    expert_attn.bias = nn.Parameter(expert_attn_bias)
    new_attention.expert_attn = expert_attn

    # Inject the pretrained output projection weights into the new MARE model
    new_attention.c_proj.weight = original_block.attn.c_proj.weight
    new_attention.c_proj.bias = original_block.attn.c_proj.bias


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
        model.transformer.h[i] = new_attention
