import torch
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
from transformers.pytorch_utils import Conv1D
from transformers.activations import ACT2FN
from typing import Optional, Tuple

from experiments.models.router import Router


class MAREMLP(GPT2MLP):
    def __init__(
        self, intermediate_size, config, num_all_rounders, num_experts
    ):
        super().__init__(intermediate_size, config)
        embed_dim = config.hidden_size
        self.num_mlps = num_all_rounders + num_experts
        self.num_all_rounders = num_all_rounders
        self.num_experts = num_experts

        self.c_fc_ar = Conv1D(intermediate_size, num_all_rounders * embed_dim)
        self.c_fc_ex = Conv1D(intermediate_size, num_experts * embed_dim)
        self.c_proj_ar = Conv1D(
            num_all_rounders * embed_dim, intermediate_size
        )
        self.c_proj_ex = Conv1D(num_experts * embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

        self.embed_dim = embed_dim

        self.router = Router(embed_dim, num_experts)

    def forward(
        self, hidden_states: Optional[Tuple[torch.FloatTensor]]
    ) -> torch.FloatTensor:
        B, T, C = hidden_states.shape
        K = T // 4

        # (B, T, E+G, C)
        hidden_states = hidden_states.repeat(1, 1, self.num_mlps).unsqueeze(-1)

        # All-rounders
        if self.num_all_rounders:
            ar_hidden_states = hidden_states[:, :, : self.num_all_rounders]
            ar_hidden_states = self.c_fc(ar_hidden_states)
            ar_hidden_states = self.act(ar_hidden_states)
            ar_hidden_states = self.c_proj(ar_hidden_states)  # (B, T, G*C)
            ar_hidden_states = ar_hidden_states.view(
                B, T, self.all_rounders, self.embed_dim
            )  # (B, T, G, C)
            ar_hidden_states = ar_hidden_states.sum(dim=2)  # (B, T, C)

        # Experts
        if self.num_experts:
            ex_hidden_states = hidden_states[:, :, self.num_all_rounders :]
            router = self.router(ex_hidden_states)  # (B, T, E)
            top_k = torch.topk(router, k=K, dim=1)  # (B, K, E)
            top_k_indices = top_k.indices
            top_k_probs = top_k.values

            # x: (B, T, E, C) -> (B, K, E, C) -> (B, K, E * C)
            ex_hidden_states = torch.gather(
                ex_hidden_states,  # (B, T, E, C)
                1,
                top_k_indices.unsqueeze(-1).expand(-1, -1, -1, C),
                # (B, K, E, C)
            ).view(
                B, K, -1
            )  # (B, K, E*C)

            ex_hidden_states = self.c_fc(ex_hidden_states)
            ex_hidden_states = self.act(ex_hidden_states)
            ex_hidden_states = self.c_proj(ex_hidden_states)  # (B, K, E*D)

            # logits: (B, T, E, C)
            ex_zeros = torch.zeros(
                B,
                T,
                self.num_experts,
                self.embed_dim,
                dtype=ex_hidden_states.dtype,
                device=ex_hidden_states.device,
            )
            ex_zeros.scatter_(
                1,
                top_k_indices.expand(-1, -1, -1, self.embed_dim),
                ex_hidden_states
                * top_k_probs.unsqueeze(-1).expand(-1, -1, -1, C),
            )
            ex_hidden_states = ex_zeros.sum(dim=2)

        if self.num_all_rounders and self.num_experts:
            hidden_states = ar_hidden_states + ex_hidden_states
        elif self.num_all_rounders:
            hidden_states = ar_hidden_states
        else:
            hidden_states = ex_hidden_states
        hidden_states = self.dropout(hidden_states)
        return hidden_states
