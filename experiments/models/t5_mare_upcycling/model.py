import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5DenseActDense
from typing import Optional, Tuple
import numpy as np

from experiments.models.router import Router


class MAREMLP(T5DenseActDense):
    def __init__(
        self,
        config,
        num_all_rounders,
        num_experts,
        capacity_factor,
    ):
        super().__init__(config)
        embed_dim = config.d_model
        intermediate_size = config.d_ff

        self.num_mlps = num_all_rounders + num_experts
        self.num_all_rounders = num_all_rounders
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        if num_all_rounders:
            self.wi_ar = nn.Linear(
                num_all_rounders * embed_dim, intermediate_size, bias=False
            )
            self.wo_ar = nn.Linear(
                intermediate_size, num_all_rounders * embed_dim, bias=False
            )
            # self.ar_router = Router(embed_dim, num_all_rounders)

        if num_experts:
            self.wi_ex = nn.Linear(
                num_experts * embed_dim, intermediate_size, bias=False
            )
            self.wo_ex = nn.Linear(
                intermediate_size, num_experts * embed_dim, bias=False
            )
            self.ex_router = Router(
                embed_dim,
                num_experts,
            )

        self.embed_dim = embed_dim

        self.scaler = nn.Sequential(nn.Linear(self.embed_dim, 1), nn.Sigmoid()) 

    def forward(
        self, hidden_states: Optional[Tuple[torch.FloatTensor]]
    ) -> torch.FloatTensor:
        B, T, C = hidden_states.shape

        # All-rounders
        if self.num_all_rounders:
            ar_hidden_states = hidden_states.repeat(
                1, 1, self.num_all_rounders
            )
            ar_hidden_states = self.wi_ar(ar_hidden_states)
            ar_hidden_states = self.act(ar_hidden_states)
            ar_hidden_states = self.dropout(ar_hidden_states)
            if (
                isinstance(self.wo_ar.weight, torch.Tensor)
                and ar_hidden_states.dtype != self.wo_ar.weight.dtype
                and self.wo_ar.weight.dtype != torch.int8
            ):
                ar_hidden_states = ar_hidden_states.to(self.wo_ar.weight.dtype)
            ar_hidden_states = self.wo_ar(ar_hidden_states)  # (B, T, G*C)
            ar_hidden_states = ar_hidden_states.view(
                B, T, self.num_all_rounders, self.embed_dim
            )  # (B, T, G, C)

            # if self.num_all_rounders > 1:
            #     ar_router = self.ar_router(hidden_states).unsqueeze(
            #         -1
            #     )  # (B, T, G, 1)
            #     ar_hidden_states = ar_hidden_states * ar_router  # (B, T, G, C)

            ar_hidden_states = ar_hidden_states.sum(dim=2)  # (B, T, C)

        # Experts
        if self.num_experts:
            router = self.ex_router(hidden_states)  # (B, T, E)
            K = (
                T
                * (self.capacity_factor - self.num_all_rounders)
                // self.num_experts
            )
            top_k = torch.topk(router, k=K, dim=1)  # (B, K, E)
            top_k_indices, indices_ = torch.sort(top_k.indices, dim=1)
            top_k_probs = torch.gather(top_k.values, 1, indices_)

            # x: (B, T, E, C) -> (B, K, E, C) -> (B, K, E * C)
            ex_hidden_states = hidden_states.repeat(1, 1, self.num_experts)
            ex_hidden_states = torch.gather(
                ex_hidden_states.view(
                    B, T, self.num_experts, -1
                ),  # (B, T, E, C)
                1,
                top_k_indices.unsqueeze(-1).expand(-1, -1, -1, C),
                # (B, K, E, C)
            ).view(
                B, K, -1
            )  # (B, K, E*C)

            ex_hidden_states = self.wi_ex(ex_hidden_states)
            ex_hidden_states = self.act(ex_hidden_states)
            ex_hidden_states = self.dropout(ex_hidden_states)
            if (
                isinstance(self.wo_ex.weight, torch.Tensor)
                and ex_hidden_states.dtype != self.wo_ex.weight.dtype
                and self.wo_ex.weight.dtype != torch.int8
            ):
                ex_hidden_states = ex_hidden_states.to(self.wo_ex.weight.dtype)
            ex_hidden_states = self.wo_ex(ex_hidden_states)  # (B, K, E*D)

            # logits: (B, T, E, C)
            ex_zeros = torch.zeros(
                B,
                T,
                self.num_experts,
                self.embed_dim,
                dtype=top_k_probs.dtype,
                device=ex_hidden_states.device,
            )

            ex_zeros.scatter_(
                1,
                top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.embed_dim),
                ex_hidden_states.view(B, K, self.num_experts, -1)
                * top_k_probs.unsqueeze(-1),
            )
            ex_hidden_states = ex_zeros.sum(dim=2)

        if self.num_all_rounders and self.num_experts:
            scale = self.scaler(hidden_states)
            hidden_states = ar_hidden_states * scale + ex_hidden_states * (1 - scale)
        elif self.num_all_rounders:
            hidden_states = ar_hidden_states
        else:
            hidden_states = ex_hidden_states

        return hidden_states
