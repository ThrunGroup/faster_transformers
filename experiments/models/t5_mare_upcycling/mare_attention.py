import torch
from torch import nn
from transformers.models.t5.modeling_t5 import (
    T5Attention,
    T5LayerSelfAttention,
    T5Attention,
)

from experiments.models.router import Router


class MAREDownCyclingSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)

        # All-Rounder Attention
        self.AllRounderSelfAttention = T5Attention(
            config, has_relative_attention_bias=False
        )
        all_rounder_dim = self.all_rounder_attention.key_value_proj_dim
        self.all_rounder_attention.q = nn.Linear(
            self.d_model, all_rounder_dim, bias=False
        )
        self.all_rounder_attention.k = nn.Linear(
            self.d_model, all_rounder_dim, bias=False
        )
        self.all_rounder_attention.v = nn.Linear(
            self.d_model, all_rounder_dim, bias=False
        )
        self.all_rounder_attention.o = nn.Linear(
            all_rounder_dim, self.d_model, bias=False
        )
        self.all_rounder_scaler = nn.Parameter(torch.zeros(8))

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
        normed_hidden_states = self.layer_norm(hidden_states)

        all_rounder_attention = self.AllRounderSelfAttention(
            normed_hidden_states,
        )

        expert_attention = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = (
            hidden_states
            + self.dropout(expert_attention[0])
            + self.dropout(all_rounder_attention[0])
            * self.all_rounder_scaler[0]
        )
        outputs = (hidden_states,) + expert_attention[
            1:
        ]  # add attentions if we output them
        return outputs


class MAREDownCyclingAttention(T5Attention):
    def __init__(
        self,
        config,
        has_relative_attention_bias=False,
        num_all_rounders: int = 0,
        num_experts: int = 0,
        capacity_factor: int = 0,
    ):
        super().__init__(config, has_relative_attention_bias)
        self.num_heads = num_all_rounders + num_experts
        self.num_all_rounders = num_all_rounders
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor - num_all_rounders
        assert self.capacity_factor >= 0, "Capacity factor must be nonnegative"

        self.router = Router(self.d_model, self.num_experts)

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        B, T, C = hidden_states.shape
        K = min(max(T // 6, 6), T)

        real_seq_length = K

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
                )
            real_seq_length += (
                past_key_value[0].shape[2]
                if query_length is None
                else query_length
            )

        key_length = (
            real_seq_length
            if key_value_states is None
            else key_value_states.shape[1]
        )

        def shape(states):
            """projection"""
            return states.view(
                B, -1, self.n_heads, self.key_value_proj_dim
            ).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return (
                states.transpose(1, 2).contiguous().view(B, -1, self.inner_dim)
            )

        def project(
            hidden_states, proj_layer, key_value_states, past_key_value
        ):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (B, n_heads, K, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (B, n_heads, K, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (B, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat(
                        [past_key_value, hidden_states], dim=2
                    )
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (B, n_heads, T, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # route
        router = self.router(hidden_states)  # (B, H, T)
        top_k = torch.topk(router, k=K, sorted=False)  # (B, H, K)
        top_k_indices, indices_ = torch.sort(top_k.indices, dim=-1)  # (B, K)
        top_k_probs = torch.gather(top_k.values * K, -1, indices_)  # (B, K)

        hidden_states = hidden_states.view(B, T, self.num_heads, self.head_dim)
        hidden_states = torch.gather(
            hidden_states,  # (B, T, H, c),
            1,
            top_k_indices.permute(0, 2, 1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, self.head_dim),  # (B, K, H, c)
        ).view(B, K, C)

        # get query states
        query_states = shape(self.q(hidden_states))  # (B, H, K, c)

        # get key/value states  (B, H, K, c)
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        # compute scores  (B, H, K, K)
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device
                )

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[
                    :, :, -hidden_states.size(1) :, :
                ]

            if mask is not None:
                position_bias = (
                    position_bias + mask
                )  # (B, n_heads, T, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (B, n_heads, K, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (B, n_heads, K, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        # attn_output = unshape(
        #     torch.matmul(attn_weights, value_states)
        # )  # (B, K, C)

        attn_output = torch.matmul(attn_weights, value_states)  # (B, H, K, c)

        # concatenate the heads
        logits = torch.zeros(
            B,
            self.num_heads,
            T,
            self.key_value_proj_dim,
            dtype=attn_output.dtype,
            device=attn_output.device,
        )
        indices = top_k_indices.unsqueeze(-1).expand(
            -1, -1, -1, self.key_value_proj_dim
        )  # (B, H, K, c)
        logits_per_head = (
            top_k_probs.unsqueeze(-1).to(attn_output.dtype) * attn_output
        )
        logits.scatter_(
            2, indices, logits_per_head
        )  # (B, H, K, c) -> (B, H, T, c)
        logits = logits.permute(0, 2, 1, 3).contiguous()  # (B, T, H, c)
        attn_output = logits.view(B, T, C)

        attn_output = self.o(attn_output)

        present_key_value_state = (
            (key_states, value_states)
            if (self.is_decoder and use_cache)
            else None
        )
        outputs = (
            (attn_output,) + (present_key_value_state,) + (position_bias,)
        )

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class PruningSelfAttention(T5Attention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.num_heads = config.num_heads
        self.pruning_scalers = nn.Parameter(torch.zeros(self.num_heads))
        self.act = nn.Sigmoid()

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
                )
            real_seq_length += (
                past_key_value[0].shape[2]
                if query_length is None
                else query_length
            )

        key_length = (
            real_seq_length
            if key_value_states is None
            else key_value_states.shape[1]
        )

        def shape(states):
            """projection"""
            return states.view(
                batch_size, -1, self.n_heads, self.key_value_proj_dim
            ).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return (
                states.transpose(1, 2)
                .contiguous()
                .view(batch_size, -1, self.inner_dim)
            )

        def project(
            hidden_states, proj_layer, key_value_states, past_key_value
        ):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat(
                        [past_key_value, hidden_states], dim=2
                    )
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(
            self.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device
                )

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[
                    :, :, -hidden_states.size(1) :, :
                ]

            if mask is not None:
                position_bias = (
                    position_bias + mask
                )  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = attn_weights * self.act(self.pruning_scalers).unsqueeze(
            -1
        ).unsqueeze(-1)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(
            torch.matmul(attn_weights, value_states)
        )  # (batch_size, seq_length, dim)

        # Softly turn off the unnecessary heads
        B, T, C = attn_output.shape
        attn_output = attn_output.view(B, T, self.num_heads, -1)
        attn_output = attn_output * self.act(self.pruning_scalers).unsqueeze(
            -1
        )
        attn_output = attn_output.view(B, T, C)

        print(
            self.act(self.pruning_scalers)[:10],
            "\n",
            self.pruning_scalers[:10],
        )

        attn_output = self.o(attn_output)

        present_key_value_state = (
            (key_states, value_states)
            if (self.is_decoder and use_cache)
            else None
        )
        outputs = (
            (attn_output,) + (present_key_value_state,) + (position_bias,)
        )

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class PruningAttention(T5LayerSelfAttention):
    def __init__(
        self,
        config,
        has_relative_attention_bias=False,
    ):
        super().__init__(config, has_relative_attention_bias)
        self.num_heads = config.num_heads
        self.pruning_scalers = nn.Parameter(torch.zeros(self.num_heads))
        self.act = nn.Sigmoid()

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
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )  # its output has (B, T, Hd) shape

        print(
            self.act(self.pruning_scalers)[:10],
            "\n",
            self.pruning_scalers[:10],
        )

        # Softly turn off the unnecessary heads
        B, T, C = attention_output[0].shape
        attention_out = attention_output[0].view(B, T, self.num_heads, -1)
        attention_out = attention_out * self.act(
            self.pruning_scalers
        ).unsqueeze(-1)
        attention_out = attention_out.view(B, T, C)

        hidden_states = hidden_states + self.dropout(attention_out)
        outputs = (hidden_states,) + attention_output[
            1:
        ]  # add attentions if we output them
        return outputs


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
        extra_attention_outputs = None

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

            if output_attentions:
                extra_attention_outputs = ar_outputs[-1]

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
        outputs = (
            (hidden_states,) + (None,) + (None,) + (extra_attention_outputs,)
        )

        return outputs
