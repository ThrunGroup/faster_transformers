import torch
from torch import nn

B, T, H = 1, 10, 4
K = 2

router = nn.Softmax(dim=1)(torch.rand(B, T, H))
top_k = torch.topk(router, k=K, dim=1, sorted=False)  # (B, K, E)

print("top k indcies: ")
print(top_k.indices)
print("top k probs: ")
print(top_k.values)

top_k_indices, indices_ = torch.sort(top_k.indices, dim=1)
top_k_probs = torch.gather(top_k.values, 1, indices_)

print("router shape: ", router.shape)
print("Router: \n", router)
print("top k indices shape: ", top_k_indices.shape)
print("top k indcies: ")
print(top_k_indices)
print("top k probs: ")
print(top_k_probs)