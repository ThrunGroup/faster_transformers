from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# class FixedRouter(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super().__init__()
#         self.fixed_router = nn.Parameter(torch.rand())
#         self.batch_norm = nn.BatchNorm1d(num_heads)
#
#
#     def forward(self, x):
#         x = self.router(x)  # (B, T, H)
#         x = x.permute(0, 2, 1)  # (B, H, T)
#         return x


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


class OracleAttention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, K:int = 10,):
        super().__init__(config, is_cross_attention, layer_idx)
        self.config = config
        self.K = K

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):

        # Diminishing token selection
        T = query.size()[-2]
        if self.layer_idx <= self.config.n_layer // 4:
            self.K = T
        elif self.layer_idx <= self.config.n_layer // 4 * 2:
            self.K = min(max(int(T / 4), 6), T)
        else:
            self.K = min(max(int(T / 6), 6), T)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [],
                value.size(-1) ** 0.5,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[
                :, :, key_length - query_length : key_length, :key_length
            ]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
                attn_weights.device
            )
            attn_weights = torch.where(
                causal_mask, attn_weights.to(attn_weights.dtype), mask_value
            )

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        top_k= torch.topk(attn_weights, self.K, dim=-1)
        attn_weights = torch.zeros_like(attn_weights).scatter(dim=-1, index=top_k.indices, src=top_k.values)
        attn_output = torch.matmul(attn_weights, value)
        with torch.no_grad():
            print(attn_weights.sum(dim=-1))

        return attn_output, attn_weights


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
        self.router = Router(self.embed_dim, self.num_heads - self.num_generalists).to(
            device
        )
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
            logits_per_head = top_k_probs.unsqueeze(-1).to(tensor.dtype) * tensor
        logits.scatter_(2, indices, logits_per_head)  # (B, H, K, c) -> (B, H, T, c)
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
        # x.permute(0, 2, 1)
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

        # project inputs to query, key, value (B, K, 3C) -> (B, K, C)
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        # split heads (B, K, C) -> (B, H, K, c)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # causal self-attention
        attn_output, attn_weights = self._attn(
            query, key, value
        )  # (B, H, K, c), (B, H, K, K)

        # put the head logits together
        attn_output = self._merge_heads(attn_output, T, top_k_indices, top_k_probs)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # HuggingFace's implementation of GPT2 requires
        # at least two outputs from the attention layer
        outputs = (attn_output, None)
        # for name, param in self.router.router.named_parameters():
        #     print("layer idx: ", self.layer_idx, f" {param.requires_grad}")
        #     print(f"{name}: {param.grad}")

        return outputs  # a, present, (attentions)


def apply_gated_attention(model, config, freeze_router: bool = True):
    # Iterate through blocks and replace self-attention layers
    for i, block in enumerate(model.transformer.h):
        new_attention = GatedAttention(config, layer_idx=i).to(device)
        new_attention.c_attn.weight = block.attn.c_attn.weight
        new_attention.c_attn.bias = block.attn.c_attn.bias
        new_attention.c_proj.weight = block.attn.c_proj.weight
        new_attention.c_proj.bias = block.attn.c_proj.bias
        block.attn = new_attention
        if freeze_router:
            for param in new_attention.router.router.parameters():
                param.requires_grad = False

def apply_oracle_attention(model, config):
    # Iterate through blocks and replace self-attention layers
    for i, block in enumerate(model.transformer.h):
        new_attention = OracleAttention(config, layer_idx=i).to(device)
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


def freeze_all_params_except_router(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, module in model.named_modules():
        print(f"{name} is regarded.")
        if "router" in name:
            for name, param in module.named_parameters():
                print(f"{name}'s grad is set to be tracked.")
                param.requires_grad = True
    for name, param in model.named_parameters():
        print(f"{name} requires grad: {param.requires_grad}.")
