import torch
from torch import nn
from transformers.models.t5.modeling_t5 import (
    T5Attention,
    T5LayerSelfAttention,
)

from experiments.models.router import Router


class MAREAttention(T5LayerSelfAttention):
    def __init__(
        self,
        config,
        has_relative_attention_bias=False,
        num_all_rounders: int = 0,
        num_experts: int = 0,
        capacity_factor: int = 0,
    ):
        super().__init__(config, has_relative_attention_bias)
        self.num_attentions = num_all_rounders + num_experts
        self.num_all_rounders = num_all_rounders
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor - num_all_rounders
        assert self.capacity_factor >= 0, "Capacity factor must be nonnegative"
        self.embed_dim = config.d_model

        self.self_attentions = nn.ModuleList(
            [
                T5Attention(
                    config,
                    has_relative_attention_bias=has_relative_attention_bias,
                )
                for _ in range(self.num_attentions)
            ]
        )

        self.expert_router = Router(self.embed_dim, self.num_experts)
        self.attention_scales = nn.Parameter(torch.ones(8))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        B, T, C = hidden_states.shape
        hidden_states = self.layer_norm(hidden_states)
        ar_hidden_states = None
        ex_hidden_states = None
        extra_attention_outputs = []

        def self_attention(index, inputs, K=T):
            mask = attention_mask[:, :, :K, :K]
            return self.self_attentions[index](
                inputs,
                mask=mask,
                position_bias=position_bias,
                layer_head_mask=layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        # All-rounders
        if self.num_all_rounders:
            ar_hidden_states = hidden_states.repeat(
                1, 1, self.num_all_rounders
            )
            ar_outputs = [
                self_attention(ar_index, ar_hidden_states)
                for ar_index in range(self.num_all_rounders)
            ]  # ARs * attention outputs (tuple)
            ar_hidden_states = torch.cat(
                [outputs[0] for outputs in ar_outputs], dim=-1
            )
            #     .sum(
            #     dim=-1
            # )  # (B, T, C)
            extra_attention_outputs.extend(
                [outputs[1:] for outputs in ar_outputs]
            )

        # Experts
        if self.num_experts:
            router = self.expert_router(hidden_states)  # (B, T, E)

            K = max((T * self.capacity_factor // self.num_experts), 1)

            top_k = torch.topk(router, k=K, dim=1)  # (B, K, E)
            top_k_indices, indices_ = torch.sort(top_k.indices, dim=1)
            top_k_probs = torch.gather(top_k.values, 1, indices_)

            # x: (B, T, E, C) -> (B, K, E, C)
            ex_hidden_states = hidden_states.repeat(1, 1, self.num_experts)
            ex_hidden_states = torch.gather(
                ex_hidden_states.view(
                    B, T, self.num_experts, -1
                ),  # (B, T, E, C)
                1,
                top_k_indices.unsqueeze(-1).expand(-1, -1, -1, C)
                # (B, K, E, C)
            )

            ex_outputs = [
                self_attention(
                    self.num_all_rounders + ex_index,
                    ex_hidden_states[:, :, ex_index],
                    K=K,
                )
                for ex_index in range(self.num_experts)
            ]  # E * attention outputs (tuple)
            ex_hidden_states = torch.cat(
                [outputs[0] for outputs in ex_outputs], dim=-1
            )  # (B, T, E*C)

            # logits: (B, T, E, C)
            ex_zeros = torch.zeros(
                B,
                T,
                self.num_experts,
                self.embed_dim,
                dtype=ex_hidden_states.dtype,
                device=ex_hidden_states.device,
            )

            ex_zeros.scatter_(
                1,
                top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.embed_dim),
                ex_hidden_states.view(B, K, self.num_experts, -1)
                * top_k_probs.unsqueeze(-1),
            )
            ex_hidden_states = ex_zeros.sum(dim=2)  # (B, T, C)

        if self.num_all_rounders and self.num_experts:
            attention_output = (
                self.attention_scales[0] * ar_hidden_states
                + self.attention_scales[1] * ex_hidden_states
            )
        elif self.num_all_rounders:
            attention_output = ar_hidden_states
        else:
            attention_output = ex_hidden_states

        hidden_states = hidden_states + self.dropout(attention_output)
        outputs = (hidden_states,) + (None,) + (None,)

        return outputs
