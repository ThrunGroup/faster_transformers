# 1. Replace the MistralMLP with mistral_direct_mlp_triton
# 2. Load the per-layer thresholds from a json file
# 3. Pass the thresholds to the MLP layers
# 4. Construct a chat-box use-case with various generation length and context length
# 5. Test the token generation speed

import torch
import torch.nn as nn
from typing import Optional, Tuple
import warnings
import inspect

import os
import json
import argparse
from kernels import mistral_mlp_sparse_direct_flag, gather_gemv2_flag, gather_transposed_gemv_flag
from transformers.activations import ACT2FN
from transformers.models.mistral.modeling_mistral import (
    MistralMLP,
    MistralRMSNorm,
    MistralDecoderLayer,
    MistralAttention,
    MistralConfig,
    apply_rotary_pos_emb,
    repeat_kv,
    _get_unpad_data,
)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MistralForCausalLM,
    MistralConfig,
)

from transformers.utils import (
    is_flash_attn_greater_or_equal_2_10,
    logging,
)

from transformers.cache_utils import Cache

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

_flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
logger = logging.get_logger(__name__)

class PredictorMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.sp_rank = config.sp_rank
        self.pred_down = nn.Linear(self.hidden_size, self.sp_rank, bias=False)
        self.pred_up = nn.Linear(self.sp_rank, self.intermediate_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]
    
    def forward(self, x):
        return  torch.abs(self.act_fn(self.pred_up(self.pred_down(x)))) > self.threshold

class FlagMistralMLP(MistralMLP):
    def __init__(self, config):
        super().__init__(config)
        self.fc2_weight_t = self.register_buffer("fc2_weigth_t", None)
    
    def forward(self, x: torch.Tensor, flags: torch.Tensor):
        if (x.shape[1] != 1):
            return super().forward(x)
        x = x.squeeze(0)
        if self.fc2_weight_t is None:
            self.fc2_weight_t = self.down_proj.weight.t().contiguous()
        x = gather_gemv2_flag(x, self.gate_proj.weight, self.up_proj.weight, flags, self.config.hidden_act)
        return gather_transposed_gemv_flag(x, self.fc2_weight_t, flags).unsqueeze(0)

class AsyncMistralFlashAttention2(MistralAttention):
    """
    Mistral flash attention module. This module inherits from `MistralAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layernorm = MistralRMSNorm(args[0].hidden_size, eps=args[0].rms_norm_eps)
        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        # self.pred_stream = torch.cuda.Stream(device="cuda")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # torch.cuda.current_stream().wait_stream(self.pred_stream)
        # with torch.cuda.stream(self.pred_stream):
        #     residual = residual.to(torch.float32).to("cpu")
        #     flags = torch.abs(torch.matmul(self.act_fn(torch.matmul(residual, self.pred_down_weight)), self.pred_up_weight) + self.pred_up_bias) > self.threshold
        #     flags = flags.to("cuda")
        down_feat = self.act_fn(torch.matmul(residual, self.pred_down_weight))
        logits = torch.matmul(down_feat, self.pred_up_weight) + self.pred_up_bias
        flags = torch.abs(logits) > self.threshold
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
            
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )
            
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)
        # torch.cuda.current_stream().wait_stream(self.pred_stream)
        if not output_attentions:
            flags = None

        return attn_output, flags, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

class PrunedMistralMLP(MistralMLP):
    def __init__(
            self,
            config,
    ):
        super().__init__(config)
        self.fc2_weight_t = self.register_buffer("fc2_weigth_t", None)
    
    def forward(self, x):
        if (x.shape[1] != 1): # batch size > 1
            return super().forward(x)# self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        x = x.squeeze(0)
        x_1 = self.act_fn(self.gate_proj(x))
        if self.fc2_weight_t is None:
            self.fc2_weight_t = self.down_proj.weight.t().contiguous()
        x_2 = mistral_mlp_sparse_direct_flag(
            x, # 0 is the beam dimension
            x_1,
            self.up_proj.weight,
            self.fc2_weight_t,
            torch.abs(x_1) > self.threshold,
        )
        return x_2.unsqueeze(0)

class AsyncSparseAsyncAttnMistralDecoderLayer(MistralDecoderLayer):
    def __init__(self, config: MistralConfig, layer_id: int):
        super().__init__(config, layer_id)
        self.mlp = FlagMistralMLP(config)
        self.self_attn = AsyncMistralFlashAttention2(config, layer_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:        
        residual = hidden_states

        hidden_states, flags, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,
            use_cache=use_cache,
        )
        
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = self.mlp(hidden_states, flags)
            
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (flags,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class AsyncSparseMistralDecoderLayer(MistralDecoderLayer):
    def __init__(self, config: MistralConfig, layer_id: int):
        super().__init__(config, layer_id)
        self.pred_mlp = PredictorMLP(config)
        self.mlp = FlagMistralMLP(config)
        self.pred_stream = torch.cuda.Stream(device="cuda")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        with torch.cuda.stream(self.pred_stream):
            flags = self.pred_mlp(residual)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        torch.cuda.current_stream().wait_stream(self.pred_stream)
        hidden_states = self.mlp(hidden_states, flags)
            
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class SparseMistralDecoderLayer(MistralDecoderLayer):
    def __init__(self, config: MistralConfig, layer_id: int):
        super().__init__(config, layer_id)
        self.pred_mlp = PredictorMLP(config)
        self.mlp = FlagMistralMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        flags = self.pred_mlp(residual)

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states, flags)
            
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def store_fake_svd_weights(model, config, path):
    svd_weights = {}
    torch.manual_seed(0)
    for i in range(len(model.model.layers)):
        svd_weights[f"layer{i}.pred_mlp.w1"] = torch.randn(config.sp_rank, config.hidden_size).to(torch.float16)
        svd_weights[f"layer{i}.pred_mlp.w2"] = torch.randn(config.intermediate_size, config.sp_rank).to(torch.float16)
        svd_weights[f"layer{i}.pred_mlp.b2"] = torch.randn(config.intermediate_size).to(torch.float16)
    torch.save(svd_weights, path)

def store_fake_svd_weights_t(model, config, path):
    svd_weights = {}
    torch.manual_seed(0)
    for i in range(len(model.model.layers)):
        svd_weights[f"layer{i}.pred_mlp.w1"] = torch.randn(config.hidden_size, config.sp_rank).to(torch.float16)
        svd_weights[f"layer{i}.pred_mlp.w2"] = torch.randn(config.sp_rank, config.intermediate_size).to(torch.float16)
        svd_weights[f"layer{i}.pred_mlp.b2"] = torch.randn(config.intermediate_size).to(torch.float16)
    torch.save(svd_weights, path)

def load_svd_weights(model, path):
    svd_weights = torch.load(path)
    for i, layer in enumerate(model.model.layers):
        layer.pred_mlp.pred_down.weight = torch.nn.Parameter(svd_weights[f"layer{i}.pred_mlp.w1"].to(torch.float16).to("cuda"))
        layer.pred_mlp.pred_up.weight = torch.nn.Parameter(svd_weights[f"layer{i}.pred_mlp.w2"].to(torch.float16).to("cuda"))
        layer.pred_mlp.pred_up.bias = torch.nn.Parameter(svd_weights[f"layer{i}.pred_mlp.b2"].to(torch.float16).to("cuda"))

def load_svd_weights_attn(model, path):
    svd_weights = torch.load(path)
    print(path)
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.pred_down_weight = svd_weights[f"layer{i}.pred_mlp.w1"].to(torch.float32)#.to(torch.float16)# .to("cuda")
        layer.self_attn.pred_up_weight = svd_weights[f"layer{i}.pred_mlp.w2"].to(torch.float32)#.to(torch.float16)# .to("cuda")
        layer.self_attn.pred_up_bias = svd_weights[f"layer{i}.pred_mlp.b2"].to(torch.float32)#.to(torch.float16)# .to("cuda")
        layer.self_attn.act_fn = ACT2FN[model.config.hidden_act]
        layer.self_attn.input_layernorm.weight = layer.input_layernorm.weight
        layer.input_layernorm.weight = None
        layer.input_layernorm = None

def transpose_weights(model):
    for layer in model.model.layers:
        assert isinstance(layer.mlp, PrunedMistralMLP) or isinstance(layer.mlp, FlagMistralMLP)
        layer.mlp.fc2_weight_t = layer.mlp.down_proj.weight.t().contiguous()

def load_thresholds(model, path):
    with open(path, "r") as f:
        thresholds = json.load(f)
    for i, layer in enumerate(model.model.layers):
        assert isinstance(layer.mlp, PrunedMistralMLP)
        layer.mlp.threshold = thresholds[str(i)]

def load_threshold_pred(model, ths_path):
    with open(ths_path, "r") as f:
        thresholds = json.load(f)
    for i, layer in enumerate(model.model.layers):
        assert isinstance(layer.pred_mlp, PredictorMLP)
        layer.pred_mlp.threshold = thresholds[str(i)]
    
def load_threshold_pred_attn(model, ths_path):
    with open(ths_path, "r") as f:
        thresholds = json.load(f)
    for i, layer in enumerate(model.model.layers):
        assert isinstance(layer.self_attn, AsyncMistralFlashAttention2)
        layer.self_attn.threshold = thresholds[str(i)]

def replace_mlp(model, config):
    for layer in model.model.layers:
        layer.mlp = PrunedMistralMLP(config).to(torch.float16).to("cuda").eval()

def replace_decoder_layer(model, config):
    new_layers = []
    for layer_id in range(config.num_hidden_layers):
        new_layers.append(SparseMistralDecoderLayer(config=config, layer_id=layer_id))
    model.model.layers = nn.ModuleList(new_layers) 

def replace_decoder_layer_async(model, config):
    new_layers = []
    for layer_id in range(config.num_hidden_layers):
        new_layers.append(AsyncSparseMistralDecoderLayer(config=config, layer_id=layer_id))
    model.model.layers = nn.ModuleList(new_layers)   

def replace_decoder_layer_async_attn(model, config):
    new_layers = []
    for layer_id in range(config.num_hidden_layers):
        new_layers.append(AsyncSparseAsyncAttnMistralDecoderLayer(config=config, layer_id=layer_id))
    model.model.layers = nn.ModuleList(new_layers)


# https://github.com/huggingface/transformers/issues/22222
def generate_sentence(tokenizer, model, prompt: str, prompt_tokens: int = 64, max_tokens: int = 128, num_return_sequences: int = 3, num_beams: int = 4):
    # generate
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").input_ids[:, :prompt_tokens].to(model.device)
    max_new_tokens = max_tokens - inputs.shape[1] - 1
    print("Input length: {}".format(inputs.shape[1]))
    print("Max new tokens: {}".format(max_new_tokens))
    assert max_new_tokens > 0, "max_tokens should be larger than the length of the prompt"
    print(inputs.shape)
    output_ids = model.generate(inputs, do_sample=True, max_new_tokens=max_new_tokens, pad_token_id=2, num_return_sequences=num_return_sequences, temperature=0.7, num_beams=num_beams)
    print(output_ids.shape)
    input_sentence = tokenizer.batch_decode(inputs, skip_special_tokens=True)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return input_sentence, outputs, output_ids.shape[-1]

def benchmark_generation(context_length, generation_length, prompt_path="text.txt", is_debugging=False, sparse_mode=0, ths_path="thresholds.json", **kwargs):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if is_debugging:
        config = MistralConfig(
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            # num_attention_heads=4,
        )
        model = MistralForCausalLM(config=config).to(torch.float16).to("cuda").eval()
    else:
        config = MistralConfig()
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
        ).to(torch.float16).to("cuda").eval()

    if sparse_mode == 0:
        replace_mlp(model, config)
        load_thresholds(model, ths_path)
        transpose_weights(model)
    elif sparse_mode == 1:
        svd_path = kwargs.pop("svd_path", "svd_weights.pth")
        config.sp_rank = kwargs.pop("sp_rank", 128)
        if not os.path.exists(svd_path):
            store_fake_svd_weights(model, config, svd_path)
        replace_decoder_layer(model, config)
        load_threshold_pred(model, ths_path)
        load_svd_weights(model, svd_path)
        model = model.to(torch.float16).to("cuda").eval()
        transpose_weights(model)
    elif sparse_mode == 2:
        svd_path = kwargs.pop("svd_path", "svd_weights.pth")
        config.sp_rank = kwargs.pop("sp_rank", 128)
        if not os.path.exists(svd_path):
            store_fake_svd_weights(model, config, svd_path)
        replace_decoder_layer_async(model, config)
        load_threshold_pred(model, ths_path)
        load_svd_weights(model, svd_path)
        model = model.to(torch.float16).to("cuda").eval()  
    elif sparse_mode == 3:
        svd_path = kwargs.pop("svd_path", "svd_weights_t.pth")
        config.sp_rank = kwargs.pop("sp_rank", 128)
        if not os.path.exists(svd_path):
            store_fake_svd_weights_t(model, config, svd_path)
        replace_decoder_layer_async_attn(model, config)
        load_threshold_pred_attn(model, ths_path)
        load_svd_weights_attn(model, svd_path)
        model = model.to(torch.float16).to("cuda").eval()

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    f = open(prompt_path, "r", encoding='utf-8')
    full_string = f.read()
    f.close()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    input_sentence, output, outlen = generate_sentence(tokenizer, model, full_string, prompt_tokens=context_length, max_tokens=generation_length, num_return_sequences=1, num_beams=1)
    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(input_sentence)
    print("=====================================")
    for i, sentence in enumerate(output):
        print(i)
        print("=====================================")
        print(sentence)
    print("=====================================")
    print("Sparse mode: {}; Context Length: {}; Generation Length: {}. Elapsed time: {} ms. Throughput: {:.2f} tokens/s".format(sparse_mode, context_length, outlen - context_length, elapsed_time_ms, outlen / (elapsed_time_ms / 1000.0)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sparse Mistral inference benchmark")
    parser.add_argument("--context_length", type=int, default=64)
    parser.add_argument("--generation_length", type=int, default=1089)
    parser.add_argument("--prompt_path", type=str, default="text.txt")
    parser.add_argument("--is_debugging", action="store_true")
    parser.add_argument("--sparse_mode", type=int, default=-1)
    parser.add_argument("--ths_path", type=str, default="thresholds.json")
    parser.add_argument("--sp_rank", type=int, default=128)
    parser.add_argument("--svd_path", type=str, default="svd_weights.pth")
    args = parser.parse_args()
    # benchmark_generation(64, 128, "text.txt", False, True, "thresholds.json")
    kwargs = {"sp_rank": args.sp_rank, "svd_path": args.svd_path}
    benchmark_generation(args.context_length, args.generation_length, args.prompt_path, args.is_debugging, args.sparse_mode, args.ths_path, kwargs=kwargs)