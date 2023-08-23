import torch
from torch import nn
from transformers import GPT2Model, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from typing import Optional, Tuple, Union

config = GPT2Config.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", config=config)


class Router(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(embed_dim, num_heads), nn.Softmax(dim=1)  # (B, T, H)
        )

    def forward(self, x):
        x = self.router(x)  # (B, T, H)
        x = x.permute(0, 2, 1)  # (B, H, T)
        return x


class GatedAttention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(
            config, is_cross_attention=is_cross_attention, layer_idx=layer_idx
        )
        self.router = Router(self.embed_dim, self.num_heads)
        self.down_sampling = nn.Linear(3 * self.embed_dim, 3 * self.head_dim)

    def _merge_heads(self, tensor, T, top_k):
        B, H, K, c = tensor.shape
        logits = torch.zeros(B, self.num_heads, T, self.head_dim)  # (B, H,
        # T, c)
        indices = top_k.indices.unsqueeze(-1).expand(
            -1, -1, -1, self.head_dim
        )  # (B, H, K, c)
        logits_per_head = top_k.values.unsqueeze(-1) * tensor
        logits.scatter_(2, indices,
                        logits_per_head)  # (B, H, K, c) -> (B, H, T, c)
        splits = logits.chunk(self.num_heads, dim=1)  # H x (B, T, c)
        out = torch.cat(splits, dim=-1).squeeze(1)  # (B, T, C)
        return out

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
        B, T, C = hidden_states.shape
        K = min(max(int(T / self.num_heads), 8), T)

        router = self.router(hidden_states)  # (B, H, T)
        top_k = torch.topk(router, k=K, dim=-1, sorted=False)  # (B, H, K)

        # pick top k tokens
        x = torch.gather(
            input=hidden_states.view(B, 1, T, C).expand(-1, self.num_heads, -1,
                                                        -1),
            dim=2,
            index=top_k.indices.view(B, self.num_heads, K, 1).expand(-1, -1,
                                                                     -1, C),
        )  # (B, H, K, C)

        # get query, key, value
        # 1. downsampling : (B, H, K, 3*C) -> (B, H, K, 3*c)
        down_sampling = self.down_sampling(self.c_attn(x))

        # 2. Divide into Q, K, V : 3 x (B, H, K, c)
        query, key, value = down_sampling.split(self.head_dim, dim=-1)

        # causal self-attention
        attn_output, attn_weights = self._attn(
            query, key, value
        )  # (B, H, K, C), (B, H, K, K)

        # put the head logits together
        attn_output = self._merge_heads(attn_output, T, top_k)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # HuggingFace's implementation of GPT2 requires
        # at least two outputs from the attention layer
        outputs = (attn_output, None)
        return outputs  # a, present, (attentions)


def get_custom_model():
    # Iterate through blocks and replace self-attention layers
    for i, block in enumerate(model.h):
        new_attention = GatedAttention(config)
        new_attention.c_attn.weight = block.attn.c_attn.weight
        new_attention.c_proj.weight = block.attn.c_proj.weight
        block.attn = new_attention
    return model


for name, param in model.named_parameters():
    print(name)
