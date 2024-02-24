from typing import Optional, Tuple, Union
import torch
import numpy as np
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from experiments.models.router import Router


class MAREAttention(GPT2Attention):
    def __init__(
        self,
        config,
        is_cross_attention=False,
        layer_idx=None,
        random_token_selection: bool = False,
        num_generalists: int = 0,
    ):
        super().__init__(
            config, is_cross_attention=is_cross_attention, layer_idx=layer_idx
        )
        self.config = config
        self.num_experts = self.num_heads - num_generalists
        self.num_generalists = num_generalists
        self.generalist_split_size = num_generalists * self.head_dim
        self.expert_split_size = self.num_experts * self.head_dim

        self.router = Router(self.embed_dim, self.num_experts)
        self.random_token_selection = random_token_selection

    def merge_heads(
        self, g_attn_output, e_attn_output, T, top_k_indices, top_k_probs
    ):
        B, _, K, c = e_attn_output.shape

        # Reshape to (B, T, G, c)
        g_attn_output = g_attn_output.permute(0, 2, 1, 3).contiguous()

        # Merge the expert logits first
        e_logits = torch.zeros(
            B,
            self.num_experts,
            T,
            self.head_dim,
            dtype=e_attn_output.dtype,
            device=e_attn_output.device,
        )
        indices = top_k_indices.unsqueeze(-1).expand(
            -1, -1, -1, self.head_dim
        )  # (B, H, K, c)
        e_logits_per_head = (
            top_k_probs.unsqueeze(-1).to(e_attn_output.dtype)
            * (e_attn_output)
            * np.sqrt(5)
        )

        e_logits.scatter_(
            2, indices, e_logits_per_head
        )  # (B, E, K, c) -> (B, E, T, c)
        e_logits = e_logits.permute(0, 2, 1, 3).contiguous()  # (B, T, E, c)

        # Concatenate the expert logits and generalist logits
        logits = torch.cat((g_attn_output, e_logits), dim=-2).view(B, T, -1)
        return logits

    def _route_tokens(self, hidden_states, K):
        # Router decides which head attends to which tokens.
        # It emits softmax outputs of shape (B, H, T).
        B, T, C = hidden_states.shape

        # extract hidden states for generalists
        generalist_hidden_states = (
            hidden_states.view(B, T, self.num_heads, self.head_dim)[
                :, :, : self.num_generalists, :
            ]
            .contiguous()
            .view(B, T, -1)
        )

        # E is the number of experts
        router = self.router(hidden_states)  # (B, E, T)
        top_k = torch.topk(router, k=K, dim=1, sorted=False)  # (B, E, K)
        # TODO (@luke): Investigate why topk sometimes outputs an arbitrary
        #  sequence and remove the following sorting
        top_k_indices, indices_ = torch.sort(top_k.indices, dim=1)
        top_k_probs = torch.gather(top_k.values, 1, indices_)

        # pick top k tokens for each Expert head
        expert_hidden_states = hidden_states.view(
            B, T, self.num_heads, self.head_dim
        )[:, :, self.num_generalists :, :]
        expert_hidden_states = torch.gather(
            expert_hidden_states,  # (B, T, E, c),
            1,
            top_k_indices.permute(0, 2, 1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, self.head_dim),  # (B, K, E, c)
        ).view(B, K, -1)

        return (
            generalist_hidden_states,
            expert_hidden_states,
            top_k_indices,
            top_k_probs,
        )

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        # select the top k tokens for each head
        B, T, C = hidden_states.shape  # (Batch size, Num tokens, Embd dim)

        # Diminishing token selection
        # K = min(
        #     max(int(T / (self.layer_idx - self.config.n_layer // 12 + 3)), 6),
        #     T,
        # )
        K = min(max(T // 5, 12), T)

        # Route the tokens to each head
        (
            generalist_hidden_states,
            expert_hidden_states,
            top_k_indices,
            top_k_probs,
        ) = self._route_tokens(hidden_states, K)

        # project inputs to query, key, value
        # generalists
        (
            generalist_query,
            generalist_key,
            generalist_value,
        ) = self.generalist_attn(generalist_hidden_states).split(
            self.generalist_split_size, dim=2
        )

        # experts
        (
            expert_query,
            expert_key,
            expert_value,
        ) = self.expert_attn(
            expert_hidden_states
        ).split(self.expert_split_size, dim=2)

        # split heads (B, K, C) -> (B, H, K, c)
        # generalists
        generalist_query = self._split_heads(
            generalist_query, self.num_generalists, self.head_dim
        )
        generalist_key = self._split_heads(
            generalist_key, self.num_generalists, self.head_dim
        )
        generalist_value = self._split_heads(
            generalist_value, self.num_generalists, self.head_dim
        )

        # experts
        expert_query = self._split_heads(
            expert_query, self.num_experts, self.head_dim
        )
        expert_key = self._split_heads(
            expert_key, self.num_experts, self.head_dim
        )
        expert_value = self._split_heads(
            expert_value, self.num_experts, self.head_dim
        )

        if layer_past is not None:
            (
                past_generalist_k,
                past_generalist_v,
                past_expert_k,
                past_expert_v,
            ) = layer_past
            generalist_key = torch.cat(
                (past_generalist_k, generalist_key), dim=-2
            )
            generalist_value = torch.cat(
                (past_generalist_v, generalist_value), dim=-2
            )
            expert_key = torch.cat((past_expert_k, expert_key), dim=-2)
            expert_value = torch.cat((past_expert_v, expert_value), dim=-2)

        if use_cache is True:
            present = (
                generalist_key,
                generalist_value,
                expert_key,
                expert_value,
            )
        else:
            present = None

        # causal self-attention
        # generalists
        g_attn_output, g_attn_weights = self._attn(
            generalist_query, generalist_key, generalist_value
        )  # (B, G, K, c), (B, G, K, K)

        # experts
        e_attn_output, e_attn_weights = self._attn(
            expert_query, expert_key, expert_value
        )  # (B, E, K, c), (B, E, K, K)

        # put the head logits together
        attn_output = self.merge_heads(
            g_attn_output, e_attn_output, T, top_k_indices, top_k_probs
        )  # (B, T, C)
        attn_output = self.c_proj(attn_output)  # output projection
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        # if output_attentions:
        #     outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
