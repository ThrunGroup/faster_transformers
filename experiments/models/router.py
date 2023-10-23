from torch import nn


class Router(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
    ):
        super().__init__()
        intermediate_size = embed_dim // 2
        self.router = nn.Sequential(
            nn.Linear(embed_dim, num_heads),  # (B, T, H)
            nn.Softmax(dim=1),  # (B, T, H)
        )

    def forward(self, x):
        x = self.router(x)  # (B, T, H)
        x = x.permute(0, 2, 1)  # (B, H, T)
        return x
