from transformers.activations import ACT2FN
from transformers.models.mistral.modeling_mistral import MistralMLP
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import gc


class SparsePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = None

    def forward(self, x):
        return self.router(x)


def low_rank_approximation(linear_layer, act_func=nn.ReLU(), rank=64):
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
    else:
        second_layer.bias.data *= 0

    sparse_predictor = SparsePredictor()
    sparse_predictor.router = nn.Sequential(first_layer, second_layer, act_func)

    return sparse_predictor
