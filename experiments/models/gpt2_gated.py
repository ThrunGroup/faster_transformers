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
    ):
        super().__init__(
            config, is_cross_attention=is_cross_attention, layer_idx=layer_idx
        )
        self.config = config
        self.num_generalists = num_generalists
        self.router = Router(
            self.embed_dim, self.num_heads - self.num_generalists
        ).to(device)
        self.softmax = nn.Softmax(dim=-1)

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
        top_k = torch.topk(router, k=K, dim=-1, sorted=False)  # (B, H, K)
        # TODO (@luke): Investigate why topk sometimes outputs an arbitrary
        #  sequence and remove the following sorting
        top_k_indices, indices_ = torch.sort(top_k.indices, dim=-1)
        top_k_probs = torch.gather(top_k.values * K, -1, indices_)

        # project inputs to query, key, value (B, T, 3Hc) -> (B, T, Hc)
        query, key, value = self.c_attn(hidden_states).split(
            self.split_size, dim=2
        )

        # split heads (B, T, C) -> (B, H, T, c)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # pick top k tokens from each QKV
        query = torch.gather(
            input=query,
            dim=2,
            index=top_k_indices.unsqueeze(-1).expand(
                -1, -1, -1, self.head_dim
            ),  # (B, H, K, c)
        )

        key = torch.gather(
            input=key,
            dim=2,
            index=top_k_indices.unsqueeze(-1).expand(
                -1, -1, -1, self.head_dim
            ),  # (B, H, K, c)
        )

        value = torch.gather(
            input=value,
            dim=2,
            index=top_k_indices.unsqueeze(-1).expand(
                -1, -1, -1, self.head_dim
            ),  # (B, H, K, c)
        )

        # causal self-attention
        attn_output, attn_weights = self._attn(
            query, key, value
        )  # (B, H, K, c), (B, H, K, K)

        # put the head logits together
        attn_output = self._merge_heads(
            attn_output, T, top_k_indices, top_k_probs
        )
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # HuggingFace's implementation of GPT2 requires
        # at least two outputs from the attention layer
        outputs = (attn_output, None)
        return outputs  # a, present, (attentions)


def apply_gated_attention(model, config):
    # Iterate through blocks and replace self-attention layers
    for i, block in enumerate(model.transformer.h):
        new_attention = GatedAttention(config, layer_idx=i).to(device)
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
