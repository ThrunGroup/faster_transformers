from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5DenseActDense
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt


class MLPRouter(nn.Module):
    def __init__(self, input_dims, output_dims, low_rank: int = 64):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(input_dims, low_rank),
            # nn.ReLU(),
            # nn.LayerNorm(low_rank),
            nn.Linear(low_rank, output_dims),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.router(x)


class MLPSparsityCheck(T5DenseActDense):
    def __init__(
            self,
            config: T5Config,
            block_idx: int,
            num_all_rounders: int = 2,
            num_experts: int = 8,
            expert_dim: int = 64,
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
        self.router = MLPRouter(
            config.d_model, config.d_ff - num_all_rounders * expert_dim, expert_dim * (num_all_rounders + num_experts)
        )
        self.router.to(self.wo.weight.dtype)

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
        ).indices.to(self.wi.weight.device)
        self.mare_mask = torch.zeros_like(self.counts_per_idx, requires_grad=False)
        self.mare_mask[self.mare_idcs] = 1

        self.non_mare_idcs = np.arange(self.config.d_ff)
        self.non_mare_idcs = torch.tensor(
            np.setdiff1d(self.non_mare_idcs, self.mare_idcs.cpu(), True), device=self.mare_idcs.device
        )
        self.is_collecting_stats = False  # Turning off collecting stats
        print(self.mare_idcs)

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
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = self.counts_per_idx.clone().detach().cpu()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.99,
            0.99,
            f"Sparsity: {round(self.num_zeros / self.num_params, 5)}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        _, _, bars = ax.hist(data, bins="auto")
        ax.set_xlabel("Counts of activations")
        ax.set_ylabel("Number of Indices")
        ax.set_title(f"{os.path.basename(filepath)}")
        ax.bar_label(
            bars,
            fontsize=5,
        )
        ax.margins(x=0.01, y=0.1)
        plt.savefig(filepath)
        plt.show()

    def forward(self, hidden_states):
        if self.is_mare is False:
            hidden_states = self.wi(hidden_states)
        else:
            # mask = torch.zeros_like(self.counts_per_idx, device=hidden_states.device)
            routing_probs = self.router(hidden_states)  # shape: ..., d_ff - num_all_rounders * expert_dim
            mask = torch.zeros(hidden_states.shape[:-1] + (self.config.d_ff, ), dtype=routing_probs.dtype, device=routing_probs.device)
            # routing_probs = torch.sum(routing_probs, dim=(0, 1))  # shape: d_ff - num_all_rounders * expert_dim
            topk_values, topk_idcs = torch.topk(
                routing_probs, k=self.num_experts * self.expert_dim, dim=-1,
            )
            topk_idcs = self.non_mare_idcs[topk_idcs]
            mask[..., self.mare_idcs] = 1
            mask = mask.scatter(-1, topk_idcs, topk_values)
            # mask[..., self.mare_idcs] = torch.finfo(mask.dtype).max
            # mask = F.sigmoid(mask)
            print(mask[...,0], torch.sum(mask[...,0]==0, dim=-1))
            hidden_states = self.wi(hidden_states) * mask
            # mask = mask.unsqueeze(-1)
            # hidden_states = F.linear(hidden_states, self.wi.weight * mask)
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

        # if self.is_mare is False:
        #     hidden_states = self.wo(hidden_states)
        # else:
        #     mask: torch.tensor
        #     hidden_states = F.linear(hidden_states, self.wo.weight * mask.T)
        return hidden_states


def activate_stats(model, init_stats: bool = False):
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
            if init_stats:
                mlp_layer.DenseReluDense.init_stats()


def plot_stats(model, filepath: str = None):
    for is_decoder, network in enumerate(
            [model.transformer.encoder, model.transformer.decoder]
    ):
        for block_idx, block in enumerate(network.block):
            if is_decoder:
                mlp_layer = block.layer[2]
            else:
                mlp_layer = block.layer[1]
            mlp_layer.DenseReluDense: MLPSparsityCheck
            mlp_layer.DenseReluDense.plot_stats(
                filepath=filepath + f"block_idx{block_idx}"
            )

def activate_mare(model):
    for is_decoder, network in enumerate(
            [model.transformer.encoder, model.transformer.decoder]
    ):
        for block_idx, block in enumerate(network.block):
            if is_decoder:
                mlp_layer = block.layer[2]
            else:
                mlp_layer = block.layer[1]
            mlp_layer.DenseReluDense: MLPSparsityCheck
            mlp_layer.DenseReluDense.activate_mare()

def apply_sparsity_check_MLP(
        model,
        config,
        num_generalists: int = 2,
        num_experts: int = 2,
        expert_size: int = 2,
):
    for is_decoder, network in enumerate(
            [model.transformer.encoder, model.transformer.decoder]
    ):
        for block_idx, block in enumerate(network.block):
            if is_decoder:
                mlp_layer = block.layer[2]
            else:
                mlp_layer = block.layer[1]
            new_mlp = MLPSparsityCheck(config, block_idx, num_all_rounders=num_generalists, num_experts=num_experts, expert_dim=expert_size)
            old_mlp = mlp_layer.DenseReluDense
            new_mlp.wi.weight.data = old_mlp.wi.weight.data
            new_mlp.wo.weight.data = old_mlp.wo.weight.data

            mlp_layer.DenseReluDense = new_mlp
    model.activate_stats = activate_stats
    model.plot_stats = plot_stats
    model.activate_mare_mlp = activate_mare
