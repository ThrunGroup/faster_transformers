from transformers import T5Config, PretrainedConfig, PreTrainedModel
from transformers.models.t5.modeling_t5 import T5DenseActDense
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import gc


class MLPRouter(nn.Module):
    def __init__(self, input_dims, output_dims, low_rank: int = 64):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(input_dims, low_rank),
            nn.Tanh(),
            nn.Linear(low_rank, output_dims),
            # nn.Hardtanh(0, 1),
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
        is_decoder: bool = True,
    ):
        super().__init__(config)
        self.config = config
        self.block_idx = block_idx
        self.num_all_rounders = num_all_rounders
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.is_mare = False
        self.is_collecting_stats = False
        self.is_decoder = is_decoder

        self.num_zeros = 0
        self.num_params = 0
        self.counts_per_idx = torch.zeros(self.config.d_ff)
        self.router = MLPRouter(
            config.d_model,
            config.d_ff - num_all_rounders * expert_dim,
            int(expert_dim * (num_all_rounders + num_experts) / 2),
        )
        self.router.to(self.wo.weight.dtype)
        self.routing_probs = None
        self.activation = None
        self.count: int = 0

        self.ar_wi = nn.Linear(*self.wi.weight.T.shape)
        self.ar_wo = nn.Linear(*self.wo.weight.T.shape)

        print("ar, exp, dim: ", num_all_rounders, num_experts, expert_dim)

    def collect_stats(self, intermediate_states):
        # Collecting stats
        self.num_zeros += int(torch.sum(intermediate_states == 0))
        self.num_params += int(torch.numel(intermediate_states))
        self.counts_per_idx = self.counts_per_idx.to(
            intermediate_states.device
        )
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
        self.mare_mask = torch.zeros_like(
            self.counts_per_idx, requires_grad=False
        )
        self.mare_mask[self.mare_idcs] = 1

        self.non_mare_idcs = np.arange(self.config.d_ff)
        self.non_mare_idcs = torch.tensor(
            np.setdiff1d(self.non_mare_idcs, self.mare_idcs.cpu(), True),
            device=self.mare_idcs.device,
        )
        self.is_collecting_stats = False  # Turning off collecting stats

        # print("self.mare_idcs: ", self.mare_idcs.shape)
        if len(self.mare_idcs) != 0:
            # self.ar_wi.weight = self.wi.weight[self.mare_idcs]
            # self.ar_wo.weight = self.wo.weight[:, self.mare_idcs]
            C, D = self.wi.weight.T.shape
            num_ars = len(self.mare_idcs)
            self.ar_wi = nn.Linear(C, num_ars).to(self.wi.weight.device)
            self.ar_wo = nn.Linear(num_ars, C).to(self.wi.weight.device)

            # print(self.ar_wi.weight.device)
        else:
            self.ar_wi = None
            self.ar_wo = None

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
        B, T, C = hidden_states.shape
        ar_hidden_states = None

        # experts
        if self.num_experts:
            K = self.num_experts * self.expert_dim

            router_probs = self.router(hidden_states)  # (B, T, 4D)
            topk_values, topk_idcs = torch.topk(
                router_probs,
                k=K,
                dim=-1,
            )  # (B, T, E)

            # get first intermediate results first
            DI, DO = self.wi.weight.T.shape

            wi_selected = torch.gather(
                self.wi.weight.T.unsqueeze(0)
                .unsqueeze(0)
                .expand(B, T, -1, -1),  # (B_, T_, DI, DO)
                -1,
                topk_idcs.unsqueeze(-2).expand(
                    -1, -1, DI, -1
                ),  # (B, T, DI_, K)
            ).view(
                B, T, -1, K
            )  # (B, T, C, K)

            hidden_states = torch.einsum(
                "btca,btce->btae", hidden_states[:, :, :, None], wi_selected
            ).reshape(B, T, K)
            hidden_states = self.act(hidden_states)
            hidden_states = hidden_states * topk_values
            #  * K  # (B, T, K)
            hidden_states = self.dropout(hidden_states)
            if (
                isinstance(self.wo.weight, torch.Tensor)
                and hidden_states.dtype != self.wo.weight.dtype
                and self.wo.weight.dtype != torch.int8
            ):
                hidden_states = hidden_states.to(self.wo.weight.dtype)

            # get final intermediates results
            DI, DO = self.wo.weight.T.shape

            wo_selected = torch.gather(
                self.wo.weight.T.unsqueeze(0)
                .unsqueeze(0)
                .expand(B, T, -1, -1),  # (B_, T_, DI, DO)
                -2,
                topk_idcs.unsqueeze(-1).expand(
                    -1, -1, -1, DO
                ),  # (B, T, K, DO_)
            ).view(
                B, T, K, -1
            )  # (B, T, K, C)

            hidden_states = torch.einsum(
                "btea,btec->btac", hidden_states[:, :, :, None], wo_selected
            ).reshape(B, T, -1)

        # all_rounders
        if hidden_states.requires_grad == True and self.num_all_rounders:
            ar_hidden_states = hidden_states @ self.ar_wi.weight.T
            ar_hidden_states = self.act(ar_hidden_states)
            ar_hidden_states = ar_hidden_states @ self.ar_wo.weight.T
            # print("ar_hidden_states: ", ar_hidden_states.shape, (B, T, C))
            if self.num_experts == 0:
                return ar_hidden_states
            else:
                hidden_states = hidden_states + ar_hidden_states

        return hidden_states


def activate_stats(model, init_stats: bool = False):
    for is_decoder, network in enumerate(
        [model.transformer.encoder, model.transformer.decoder]
    ):
        for block_idx, block in enumerate(network.block):
            if is_decoder:
                mlp_layer = block.layer[2]
                # continue
            else:
                if block_idx < 6:
                    continue
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
                if block_idx < 6:
                    continue
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
                # continue
            else:
                if block_idx < 6:
                    continue
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
                # continue
            else:
                if block_idx < 6:
                    continue
                mlp_layer = block.layer[1]
            new_mlp = MLPSparsityCheck(
                config,
                block_idx,
                num_all_rounders=num_generalists,
                num_experts=num_experts,
                expert_dim=expert_size,
                is_decoder=bool(is_decoder),
            )
            old_mlp = mlp_layer.DenseReluDense
            new_mlp.wi.weight.data = old_mlp.wi.weight.data
            new_mlp.wo.weight.data = old_mlp.wo.weight.data

            mlp_layer.DenseReluDense = new_mlp
            del old_mlp
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()

    model.activate_stats = activate_stats
    model.plot_stats = plot_stats
    model.activate_mare_mlp = activate_mare
