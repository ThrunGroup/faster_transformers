from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class Router(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(embed_dim, num_heads),  # (B, T, H)
            nn.Softmax(dim=1),
        )
        self.batch_norm = nn.BatchNorm1d(num_heads)

    def forward(self, x):
        x = self.router(x)  # (B, T, H)
        x = x.permute(0, 2, 1)  # (B, H, T)
        return x


class GatedAttention(GPT2Attention):
    def __init__(
        self,
        config,
        is_cross_attention=False,
        layer_idx=None,
        num_generalists: int = 0,
        random_token_selection: bool = False,
    ):
        super().__init__(
            config, is_cross_attention=is_cross_attention, layer_idx=layer_idx
        )
        self.config = config
        self.num_generalists = num_generalists
        self.router = Router(
            self.embed_dim, self.num_heads - self.num_generalists
        )
        self.random_token_selection = random_token_selection

    def merge_heads(self, tensor, T, K, top_k_indices, top_k_probs):
        if K == T:
            attn_output = self._merge_heads(
                tensor, self.num_heads, self.head_dim
            )
        else:
            attn_output = self._merge_gated_heads(
                tensor, T, top_k_indices, top_k_probs
            )
        return attn_output

    def _merge_gated_heads(self, tensor, T, top_k_indices, top_k_probs):
        B, H, K, c = tensor.shape
        logits = torch.zeros(
            B,
            self.num_heads,
            T,
            self.head_dim,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        indices = top_k_indices.unsqueeze(-1).expand(
            -1, -1, -1, self.head_dim
        )  # (B, H, K, c)
        if self.random_token_selection or self.layer_idx == 0:
            # or self.layer_idx <= self.config.n_layer // 4:
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

    def _route_tokens(self, hidden_states, K, layer_past):
        # Router decides which head attends to which tokens.
        # It emits softmax outputs of shape (B, H, T).
        B, T, C = hidden_states.shape

        if T == K or layer_past is not None:
            return hidden_states, None, None

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
        ).view(B, K, C)
        return hidden_states, top_k_indices, top_k_probs

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
        K = min(max(T // 2, 12), T)
        # if self.layer_idx <= self.config.n_layer // 12:
        #     K = T
        # else:
        #     K = min(max(int(T / (self.layer_idx - self.config.n_layer // 12 + 2)), 6), T)
        # elif self.layer_idx <= self.config.n_layer // 4 * 2:
        #     K = min(max(int(T / 4), 6), T)
        # elif self.layer_idx <= self.config.n_layer // 4 * 3:
        #     K = min(max(int(T / 8), 6), T)
        # else:
        #     K = min(max(int(T / 12), 6), T)

        # Route the tokens to each head
        hidden_states, top_k_indices, top_k_probs = self._route_tokens(
            hidden_states, K, layer_past
        )

        # project inputs to query, key, value (B, K, 3C) -> (B, K, C)
        query, key, value = self.c_attn(hidden_states).split(
            self.split_size, dim=2
        )

        # split heads (B, K, C) -> (B, H, K, c)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # causal self-attention
        attn_output, attn_weights = self._attn(
            query, key, value
        )  # (B, H, K, c), (B, H, K, K)

        # put the head logits together
        attn_output = self.merge_heads(
            attn_output, T, K, top_k_indices, top_k_probs
        )
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # HuggingFace's implementation of GPT2 requires
        # at least two outputs from the attention layer
        outputs = (attn_output, present)
        return outputs  # a, present, (attentions)


def apply_gated_attention(
    model,
    config,
    random_token_selection,
    num_generalists: int = 0,
):
    # Iterate through blocks and replace self-attention layers
    for i, block in enumerate(model.transformer.h):
        new_attention = GatedAttention(
            config, layer_idx=i, random_token_selection=random_token_selection
        )
        new_attention.c_attn.weight = block.attn.c_attn.weight
        new_attention.c_attn.bias = block.attn.c_attn.bias
        new_attention.c_proj.weight = block.attn.c_proj.weight
        new_attention.c_proj.bias = block.attn.c_proj.bias
        block.attn = new_attention


def freeze_model(model):
    blocks_to_train = [f"h.{i}." for i in list(range(6, 12))] + ["score"]
    layers_to_train = ["c_proj", "router", "ln_f"] + blocks_to_train

    # Freeze the specified layers
    for name, param in model.named_parameters():
        do_train = any([layer in name for layer in layers_to_train])
        if not do_train:
            param.requires_grad = False
