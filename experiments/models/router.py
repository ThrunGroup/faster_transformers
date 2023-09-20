from torch import nn


class Router(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(embed_dim, num_heads),  # (B, T, H)
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.router(x)  # (B, T, H)
        # x = x.permute(0, 2, 1)  # (B, H, T)
        return x
