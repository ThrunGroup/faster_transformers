from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5DenseActDense
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

from experiments.models.router import Router


class MLPSparsityCheck(T5DenseActDense):
    def __init__(
        self,
        config: T5Config,
        block_idx: int,
        num_all_rounders: int = 1,
        num_experts: int = 4,
        expert_dim: int = 16,
    ):
        super().__init__(config)
        self.config = config
        self.block_idx = block_idx
        self.num_all_rounders = num_all_rounders
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.is_mare = False
        self.is_collecting_stats = False

        self.num_zeros = 0
        self.num_params = 0
        self.counts_per_idx = torch.zeros(config.d_ff)
        self.router = Router(config.d_model, config.d_ff)

    def collect_stats(self, intermediate_states):
        # Collecting stats
        self.num_zeros += int(torch.sum(intermediate_states == 0))
        self.num_params += int(torch.numel(intermediate_states))
        self.counts_per_idx = self.counts_per_idx.to(intermediate_states.device)
        self.counts_per_idx += torch.sum(intermediate_states != 0, dim=(0, 1))

    def init_stats(self):
        self.num_zeros = 0
        self.num_params = 0
        self.counts_per_idx = torch.zeros(self.config.d_ff)

    def activate_mare(self):
        self.is_mare = True
        self.mare_idcs = torch.topk(
            self.counts_per_idx, k=self.num_all_rounders * self.expert_dim
        ).indices

    def deactivate_mare(self):
        self.is_mare = False

    def activate_stats(self):
        self.is_collecting_stats = True

    def deactivate_stats(self):
        self.is_collecting_stats = False

    def plot_stats(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "mare_mlp_histograms",
                f"block{self.block_idx}_histogram",
            )
        data = self.counts_per_idx.clone().detach().cpu()
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins="auto")
        plt.text(
            1,
            1,
            f"Sparsity: {self.num_zeros/self.num_params}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.savefig(filepath)
        plt.show()

    def forward(self, hidden_states):
        if self.is_mare is False:
            hidden_states = self.wi(hidden_states)
        else:
            hidden_states = F.linear(hidden_states, self.wi)
        hidden_states = self.act(hidden_states)

        # Collecting stats
        if self.is_collecting_stats is True:
            self.collect_stats(hidden_states)
        # print(f"block idx: {self.block_idx} / sparsity:{self.num_zeros/self.num_params}")
        # sorted_values, sorted_idcs = self.counts_per_idx.clone().detach().sort()
        # # print(torch.histogram(self.counts_per_idx.to('cpu'), 32, density=True))
        # # print(f"block idx: {self.block_idx}/ sorted counts: {sorted_values[:5], sorted_values[-5:]}/"
        # #       f"sorted idcs: {sorted_idcs[:5], sorted_idcs[-5:]}")
        # # print(f"block idx: {self.block_idx} / sparsity:{self.num_negatives / self.num_params}")
        # # print(f"block idx: {self.block_idx} / sparsity:{self.num_positives / self.num_params}")
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


def activate_stats(model):
    for is_decoder, network in enumerate(
        [model.transformer.encoder, model.transformer.decoder]
    ):
        for block_idx, block in enumerate(network.block):
            if is_decoder:
                mlp_layer = block.layer[2]
            else:
                mlp_layer = block.layer[1]
            mlp_layer.DenseReluDense: MLPSparsityCheck
            mlp_layer.DenseReluDense.activate_stats()


def plot_stats(model):
    for is_decoder, network in enumerate(
        [model.transformer.encoder, model.transformer.decoder]
    ):
        for block_idx, block in enumerate(network.block):
            if is_decoder:
                mlp_layer = block.layer[2]
            else:
                mlp_layer = block.layer[1]
            mlp_layer.DenseReluDense: MLPSparsityCheck
            mlp_layer.DenseReluDense.plot_stats()


def apply_sparsity_check_MLP(
    model,
    config,
):
    for is_decoder, network in enumerate(
        [model.transformer.encoder, model.transformer.decoder]
    ):
        for block_idx, block in enumerate(network.block):
            if is_decoder:
                mlp_layer = block.layer[2]
            else:
                mlp_layer = block.layer[1]
            new_mlp = MLPSparsityCheck(config, block_idx)
            old_mlp = mlp_layer.DenseReluDense
            new_mlp.wi.weight.data = old_mlp.wi.weight.data
            new_mlp.wo.weight.data = old_mlp.wo.weight.data

            mlp_layer.DenseReluDense = new_mlp
    model.activate_stats = activate_stats
    model.plot_stats = plot_stats
