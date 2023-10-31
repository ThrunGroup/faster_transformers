import torch
import torch.nn as nn


def low_rank_approximation(linear_layer, rank=64):
    # Decompose the weight matrix of the layer
    U, S, V = torch.svd(linear_layer.weight.T.data)

    print(U.shape, S.shape, V.shape)

    # Take the top `rank` components
    U_approx = U[:, :rank]
    S_approx = torch.diag(S[:rank])
    V_approx = V[:, :rank].t()

    # Create two new linear layers for low-rank approximation
    first_layer = nn.Linear(linear_layer.in_features, rank, bias=False)
    second_layer = nn.Linear(rank, linear_layer.out_features, bias=True)

    # Assign the low-rank matrices to the layers' weights
    first_layer.weight.data = U_approx.T
    second_layer.weight.data = torch.mm(S_approx, V_approx).T

    # If the original linear layer had a bias, assign it to the second layer's bias
    if linear_layer.bias is not None:
        second_layer.bias.data = linear_layer.bias.data

    return first_layer, second_layer


# Sample usage
wi = nn.Linear(512, 1024)
low_rank_wi_1, low_rank_wi_2 = low_rank_approximation(wi, rank=400)

print("wi: ", wi.weight.T.shape)
print(low_rank_wi_1.weight.T.shape)
print(low_rank_wi_2.weight.T.shape)

new = nn.Sequential(low_rank_wi_1, low_rank_wi_2)

x = torch.ones((16, 512))

print(wi(x), new(x))
