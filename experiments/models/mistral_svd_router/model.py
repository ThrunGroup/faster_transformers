from transformers.activations import ACT2FN
from transformers.models.mistral.modeling_mistral import MistralMLP
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
            nn.ReLU(),
        )

    def forward(self, x):
        return self.router(x)


def activate_stats(model, init_stats: bool = False):
    for is_decoder, network in enumerate(
        [model.transformer.encoder, model.transformer.decoder]
    ):
        for block_idx, block in enumerate(network.block):
            if is_decoder:
                mlp_layer = block.layer[2]
                # continue
            else:
                # if block_idx < 6:
                #     continue
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
                # if block_idx < 6:
                #     continue
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
                # if block_idx < 6:
                #     continue
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
                # if block_idx < 6:
                #     continue
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

            # router
            new_mlp.router.router = low_rank_approximation(
                old_mlp.wi,
                # rank=expert_size * (num_experts + num_generalists),
                # rank=old_mlp.ㅑ.out_features // 5,
                rank=256,
            )

            if block_idx == 0:
                print(
                    "RANK: ",
                    expert_size * (num_experts + num_generalists) // 2,
                )

            mlp_layer.DenseReluDense = new_mlp
            del old_mlp
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()

    model.activate_stats = activate_stats
    model.plot_stats = plot_stats
    model.activate_mare_mlp = activate_mare


def low_rank_approximation(linear_layer, rank=64):
    # Decompose the weight matrix of the layer
    U, S, V = torch.svd(linear_layer.weight.T.data)

    # Take the top `rank` components
    U_approx = U[:, :rank]
    S_approx = torch.diag(S[:rank])
    V_approx = V[:, :rank].t()

    # Create two new linear layers for low-rank approximation
    first_layer = nn.Linear(linear_layer.in_features, rank, bias=False)
    second_layer = nn.Linear(rank, linear_layer.out_features, bias=True)

    # Assign the low-rank matrices to the layers' weights
    first_layer.weight.data = U_approx.T.contiguous()
    second_layer.weight.data = torch.mm(S_approx, V_approx).T.contiguous()

    # If the original linear layer had a bias, assign it to the second layer's bias
    if linear_layer.bias is not None:
        second_layer.bias.data = linear_layer.bias.data

    return nn.Sequential(first_layer, second_layer, nn.ReLU())
