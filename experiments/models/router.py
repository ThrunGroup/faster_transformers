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
            # nn.Linear(embed_dim, intermediate_size),
            # nn.GELU(),
            # nn.Linear(intermediate_size, num_heads),  # (B, T, H)
            nn.Linear(embed_dim, num_heads),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.router(x)  # (B, T, H)
        # x = x.permute(0, 2, 1)  # (B, H, T)
        return x
