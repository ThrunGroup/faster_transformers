from typing import Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)
from router import Router

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class GatedAttention(LlamaAttention):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.num_layers = config.n_layer
        self.router = Router(self.embed_dim, self.num_heads).to(device)

        print("pretraining_tp: ", self.config.pretraining_tp)

    def _reformat(self, tensor, bsz, q_len):
        return tensor.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def _merge_heads(self, tensor, T, top_k_indices, top_k_probs):
        B, H, K, c = tensor.shape
        logits = torch.zeros(
            B,
            self.num_heads,
            T,
            self.head_dim,
            dtype=tensor.dtype,
            device=device,
        )
        indices = top_k_indices.unsqueeze(-1).expand(
            -1, -1, -1, self.head_dim
        )  # (B, H, K, c)
        if self.layer_idx <= self.config.n_layer // 4:
            logits_per_head = tensor
        else:
            logits_per_head = (
                top_k_probs.unsqueeze(-1).to(tensor.dtype) * tensor
            )
        logits.scatter_(
            2, indices, logits_per_head
        )  # (B, H, K, c) -> (B, H, T, c)
        logits = logits.permute(0, 2, 1, 3).contiguous()  # (B, T, H, c)
        logits = logits.view(B, T, -1)
        return logits

    def _route_tokens(
        self,
        hidden_states,
    ):
        B, T, C = hidden_states.size()
        # Router decides which head attends to which tokens.
        # It emits softmax outputs of shape (B, H, T).

        # Diminishing token selection
        if self.layer_idx <= self.config.n_layer // 4:
            K = T
        elif self.layer_idx <= self.config.n_layer // 4 * 2:
            K = min(max(int(T / 4), 6), T)
        else:
            K = min(max(int(T / 6), 6), T)

        # G is the number of generalists
        router = self.router(hidden_states)  # (B, H, T)
        top_k = torch.topk(router, k=K, sorted=False)  # (B, H, K)
        # TODO (@luke): Investigate why topk sometimes outputs an arbitrary
        #  sequence and remove the following sorting
        top_k_indices, indices_ = torch.sort(top_k.indices, dim=-1)
        top_k_probs = torch.gather(top_k.values * K, -1, indices_)

        hidden_states = hidden_states.view(B, T, self.num_heads, self.head_dim)
        hidden_states = torch.gather(
            hidden_states,  # (B, T, H, c),
            1,
            top_k_indices.permute(0, 2, 1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, self.head_dim),  # (B, K, H, c)
        ).view(
            B, K, C
        )  # (B, K, C)

        return hidden_states, top_k_indices, top_k_probs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        """
        In the paper, LLama2 used a new MHA architecture called GHA,
        a generalization of multi-query attention which uses an intermediate
        (more than one, less than number of query heads) number of
        key-value heads.

        However, the pretrained model available on HuggingFace uses the usual
        MHA. So,
        num_key_value_heads = num_heads by default
        num_key_value_groups = 1 by default
        """
        B, T, C = hidden_states.size()

        # Router decides which head attends to which tokens.
        # It emits softmax outputs of shape (B, H, T).
        hidden_states, top_k_indices, top_k_probs = self._route_tokens(
            hidden_states
        )

        # Q, K, V projection
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reformat to (B, H, T, c)
        query_states = self._reformat(query_states)
        key_states = self._reformat(key_states)
        value_states = self._reformat(value_states)

        kv_seq_len = key_states.shape[-2]  # Just num heads (H)
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        # @luke: but again, this model uses MHA, so nothing is repeated here.
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Self-attention here! N^2 complexity
        # Output shape:(B, H, T, T)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / np.sqrt(self.head_dim)

        if attn_weights.size() != (B, self.num_heads, T, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size "
                f"{(B, self.num_heads, T, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (B, 1, T, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size "
                    f"{(B, 1, T, kv_seq_len)}, "
                    f"but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (B, self.num_heads, T, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(B, self.num_heads, T, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # Merge heads
        # attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self._merge_heads(
            attn_output, T, top_k_indices, top_k_probs
        )

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def apply_gated_attention(model, config):
    # Iterate through blocks and replace self-attention layers
    for i, block in enumerate(model.transformer.h):
        new_attention = GatedAttention(config, layer_idx=i).to(device)
        new_attention.c_attn.weight = block.attn.c_attn.weight
        new_attention.c_attn.bias = block.attn.c_attn.bias
        new_attention.c_proj.weight = block.attn.c_proj.weight
        new_attention.c_proj.bias = block.attn.c_proj.bias
        block.attn = new_attention


if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b"
    model = LlamaModel.from_pretrained(model_name)

    print("Model: ")
    print(model)

    print("Config: ")
    print(model.config)
