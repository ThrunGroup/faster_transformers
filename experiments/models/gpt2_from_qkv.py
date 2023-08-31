import torch
from torch import nn
from transformers import GPT2Model, GPT2Config, GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from typing import Optional, Tuple, Union


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

    def _merge_heads(self, tensor, T, top_k):
        B, H, K, c = tensor.shape
        logits = torch.zeros(B, self.num_heads, T, self.head_dim)
        # indices: (B, H, K, c)
        indices = top_k.indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
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

        # get query, key, value
        attention_proj = self.c_attn(hidden_states)  # (B, T, 3C)

        # (B, T, 3C) -> (B, H, T, 3c)
        attention_proj = attention_proj.view(B,
                                             T,
                                             self.num_heads,
                                             3 * self.head_dim).permute(0, 2,
                                                                        1, 3)

        # pick top k tokens # (B, H, K, C)
        router = self.router(hidden_states)  # (B, H, T)
        top_k = torch.topk(router, k=K, dim=-1, sorted=False)  # (B, H, K)

        # (B, H, K, 3c)
        attention_per_head = torch.gather(
            input=attention_proj,  # (B, H, T, 3c)
            dim=2,
            index=top_k.indices.unsqueeze(-1).expand(-1, -1, -1,
                                                     3 * self.head_dim),
            # (B, H, K, 3C)
        )

        # (B, H, K, c)
        query, key, value = attention_per_head.split(self.head_dim, dim=-1)

        # causal self-attention
        attn_output, attn_weights = self._attn(
            query, key, value
        )  # (B, H, K, c), (B, H, K, K)

        # put the head logits together
        attn_output = self._merge_heads(attn_output, T, top_k)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # HuggingFace's implementation of GPT2 requires
        # at least two outputs from the attention layer
        outputs = (attn_output, None)
        return outputs  # a, present, (attentions)


def apply_gated_attention(model, config):
    # Iterate through blocks and replace self-attention layers
    for i, block in enumerate(model.h):
        new_attention = GatedAttention(config)
        new_attention.c_attn.weight = block.attn.c_attn.weight
        new_attention.c_proj.weight = block.attn.c_proj.weight
        block.attn = new_attention


def freeze_model(model):
    blocks_to_train = [f"h.{i}." for i in list(range(6, 12))]
    layers_to_train = ["c_proj", "router", "ln_f"] + blocks_to_train

    # Freeze the specified layers
    for name, param in model.named_parameters():
        do_train = any([layer in name for layer in layers_to_train])
        if not do_train:
            param.requires_grad = False


def build_gated_attention_model():
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", config=config)

    apply_gated_attention(model, config)
    freeze_model(model)

    return model


if __name__ == '__main__':
    build_gated_attention_model()
