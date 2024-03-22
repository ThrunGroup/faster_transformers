from transformers import TrainerCallback, Trainer
from peft import PeftModel
from datasets import Dataset
from transformers.utils import is_sagemaker_mp_enabled, is_sagemaker_dp_enabled
from typing import Any, Dict, Union, Optional, Tuple
from torch.nn import MSELoss

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy

from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaConfig,
    LlamaForCausalLM,
)
from experiments.models.sparse_mistral.svd_router import (
    low_rank_approximation,
)
from experiments.models.sparse_silu.utils import apply_sparse_silu_mlp, apply_sparse_decoder_layer, SparseSiLU


class SparseLlamaConfig(LlamaConfig):
    model_type = "sparse_llama"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SparseLlamaForCausalLM(LlamaForCausalLM):
    config_class = SparseLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if config.use_sparse_model:
            self.apply_sparse_mlp()
            if config.thresholds is not None:
                for idx, m in enumerate(self.model.layers):
                    if isinstance(m.mlp, LlamaSparseSiluMLP):
                        m.mlp.dead_threshold = config.thresholds[idx]
                        m.mlp.sparse_act_fn.set_new_threshold(m.mlp.dead_threshold)
                        m.mlp.kill_sparse_swish_outputs = True
                        m.mlp.use_relu = config.use_relu
        if config.use_sparse_predictor:
            self.apply_sparse_predictor(init_svd=config.init_svd)

    def apply_sparse_mlp(self):
        apply_sparse_silu_mlp(
            self,
            config=self.config,
            use_sparse_regularization=self.config.use_sparse_regularization,
        )

    def apply_sparse_predictor(self, init_svd: bool = True):
        apply_sparse_decoder_layer(self, config=self.config, init_svd=init_svd)


class LlamaSparseSiluMLP(LlamaMLP):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.swish_outputs = None
        self.relu = nn.ReLU()

        self.kill_sparse_swish_outputs = False
        self.dead_percentage = 0
        self.is_stats = False
        self.visit_counts = 0

        # Hyperparameters to tune
        self.dead_threshold = kwargs.pop("dead_threshold", 0)
        self.use_sparse_regularization = kwargs.pop("use_sparse_regularization", True)
        self.regularization_type = kwargs.pop("regularization_type", "L1 regularization")
        self.regularization_threshold = kwargs.pop("regularization_threshold", 0.5)
        self.use_relu = kwargs.pop("use_relu", False)
        self.activation_norm = None

        # Activation Histograms
        self.is_collect_histogram = False
        num_bins = 1000
        self.histogram_bins = torch.linspace(-1, 1, num_bins - 2)
        self.histogram_bins = torch.cat([torch.tensor([-torch.inf]), self.histogram_bins, torch.tensor([torch.inf])])
        self.pre_act_hist_counts = torch.zeros(num_bins - 1)
        self.post_act_hist_counts = torch.zeros(num_bins - 1)
        self.t = 0
        self.count = 0
        self.agg_sparsity = 0

        # Sparse activation function
        self.sparse_act_fn = SparseSiLU(threshold=self.dead_threshold)

    def activate_stats(self, is_collect_histogram: bool = True):
        self.is_stats = True
        self.dead_percentage = 0
        self.visit_counts = 0
        self.is_collect_histogram = is_collect_histogram
        self.histogram_counts = torch.zeros(2000)  # .to(self.down_proj.weight.device)

    def deactivate_stats(self):
        self.is_stats = False

    def collect_stats(self, pre_activation, post_activation):
        start_time = time.time()
        pre_activation = pre_activation.float().cpu().detach()
        post_activation = post_activation.float().cpu().detach()
        # self.histogram_bins=self.histogram_bins.to(pre_activation.device).type(pre_activation.dtype)
        self.pre_act_hist_counts += torch.histogram(pre_activation, bins=self.histogram_bins)[0]
        self.post_act_hist_counts += torch.histogram(torch.abs(post_activation), bins=self.histogram_bins)[0]
        self.t += time.time() - start_time
        if self.visit_counts % 30 == 0:
            print(f"Time taken to collect stats: {self.t}s.")

    def forward(
        self,
        x,
        sp_mask: torch.tensor = None,
    ):
        """
        If kill_sparse_swish_outputs is set to False, this layer functions exactly like a normal MLP layer.
        """
        if sp_mask != None:  # When sparse mask is given
            return self.down_proj(
                self.sparse_act_fn(self.gate_proj(x) * sp_mask) * self.up_proj(x)
            )  # Todo: This doesn't accelerate runtime (instead slowing down)

        elif self.use_relu:
            post_act = self.relu(self.gate_proj(x))
            self.count += 1
            if self.count <= 1:
                print("USING RELU!!!!")

            if self.is_stats:
                dead_neurons = post_act == 0
                dead_percentage = dead_neurons.float().mean()
                agg_sparsity = dead_neurons.all(dim=0).float().mean()

                self.dead_percentage = (self.dead_percentage * self.visit_counts + dead_percentage) / (self.visit_counts + 1)
                self.agg_sparsity = (self.agg_sparsity * self.visit_counts + agg_sparsity) / (self.visit_counts + 1)
                self.visit_counts += 1

            return self.down_proj(post_act * self.up_proj(x))

        else:
            self.count += 1
            if self.count <= 1:
                print("USING SparseSILU!!!!")
            pre_act = self.gate_proj(x)
            post_act = self.act_fn(pre_act)
            if self.kill_sparse_swish_outputs:
                dead_neurons = post_act.abs() <= self.dead_threshold
                # print("pre act sparsity: ", (pre_act==0).float().mean())

                dead_percentage = dead_neurons.float().mean()
                agg_sparsity = dead_neurons.all(dim=0).float().mean()

                if self.is_stats:
                    self.dead_percentage = (self.dead_percentage * self.visit_counts + dead_percentage) / (self.visit_counts + 1)
                    self.agg_sparsity = (self.agg_sparsity * self.visit_counts + agg_sparsity) / (self.visit_counts + 1)
                    self.visit_counts += 1

                    self.a = dead_percentage

                    # Collect histogram stats
                    if self.is_collect_histogram and pre_act.eq(0).float().mean() < 0.99:  # Padded dataset
                        self.collect_stats(pre_act, post_act)

                if self.count <= 1:
                    print("KILL!")
                post_act[dead_neurons] = 0

            out = self.down_proj(post_act * self.up_proj(x))
            if self.use_sparse_regularization:
                if self.regularization_type == "L1 regularization":
                    self.activation_norm = torch.abs(post_act)[torch.abs(post_act) < self.regularization_threshold].mean()
                elif self.regularization_type == "L2 regularization":
                    self.activation_norm = torch.sqrt(torch.square(post_act)[torch.abs(post_act) < self.regularization_threshold]).mean()

            return out


class LlamaSparseDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        decoder_layer: LlamaDecoderLayer,
        init_svd: bool = True,
        *args,
        **kwargs,
    ):
        assert isinstance(decoder_layer.mlp, LlamaSparseSiluMLP), f"{type(decoder_layer.mlp)} should be LlamaSparseSiluMLP."

        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.init_svd = init_svd
        self.self_attn = decoder_layer.self_attn

        self.mlp = decoder_layer.mlp
        self.input_layernorm = decoder_layer.input_layernorm
        self.post_attention_layernorm = decoder_layer.post_attention_layernorm

        # Sparse predictor for mlp (initialized with SVD decomposed matrix)
        self.low_rank = kwargs.pop("low_rank", 64)
        self.sparse_act_func = decoder_layer.mlp.sparse_act_fn

        print(f"Setting {layer_idx}th mlp layer's sparse predictor... svd init: {init_svd}")
        self.sp_mlp = low_rank_approximation(
            decoder_layer.mlp.gate_proj,
            act_func=self.sparse_act_func,
            init_svd=init_svd,
        )
        self.use_async = kwargs.pop("use_async", False)
        self.use_sparse_predictor = False
        self.distill_loss = None

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
        print("hidden_states shape: ", hidden_states.shape)
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states
        sp_mask = None

        if self.use_async:
            sp_mask = self.sp_mlp(hidden_states)

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if not self.use_async:
            sp_mask = self.sp_mlp(hidden_states)

        # Compute distillation loss
        gating_output = self.mlp.sparse_act_fn(self.mlp.gate_proj(hidden_states))
        loss_func = MSELoss()
        self.distill_loss = loss_func(sp_mask, gating_output)

        # Convert sp mask into binary form
        sp_mask = sp_mask > 0

        if self.training:
            sp_mask = None
        # if not self.use_sparse_predictor:
        #     sp_mask = None

        hidden_states = self.mlp(hidden_states, sp_mask)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
