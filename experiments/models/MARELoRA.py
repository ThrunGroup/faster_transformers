import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class Router(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(embed_dim, num_heads),  # (B, T, H)
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.router(x)  # (B, T, H)
        return x


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        capacity_factor: int = 2,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = True,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        num_experts: int = 0,
        num_all_rounders: int = 0,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_all_rounder_A = nn.Parameter(
                self.weight.new_zeros((r, in_features))
            )
            self.lora_all_rounder_B = nn.Parameter(
                self.weight.new_zeros((out_features, r))
            )
            self.lora_experts_A = nn.Parameter(
                self.weight.new_zeros(
                    (num_experts * r, num_experts * in_features)
                )
            )
            self.lora_experts_B = nn.Parameter(
                self.weight.new_zeros(
                    (num_experts * out_features, num_experts * r)
                )
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

        # MoE
        self.router = Router(in_features, num_experts)
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts
        self.out_features = out_features
        self.num_all_rounders = num_all_rounders

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_all_rounder_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_all_rounder_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_all_rounder_B)

        if hasattr(self, "lora_experts_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_experts_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_experts_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (
                        T(self.lora_all_rounder_B @ self.lora_all_rounder_A)
                        + T(self.lora_experts_B @ self.lora_experts_A)
                    ) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(
                        self.lora_all_rounder_B @ self.lora_all_rounder_A
                    ) + T(self.lora_experts_B @ self.lora_experts_A)
                self.merged = True

    def moe_forward(self, x: torch.Tensor):
        B, T, C = x.shape

        K = T // 4

        router = self.router(x)  # (B, T, C) -> (B, T, E)
        top_k = torch.topk(router, k=K, dim=1)  # (B, K, E)
        top_k_indices, indices_ = torch.sort(top_k.indices, dim=-1)
        top_k_indices = top_k_indices.unsqueeze(-1)
        top_k_probs = torch.gather(top_k.values, -1, indices_)

        # x: (B, T, C) -> (B, T, E, C) -> (B, K, E, C) -> (B, K, E * C)
        x = torch.gather(
            x.unsqueeze(2).expand(
                -1, -1, self.num_experts, -1
            ),  # (B, T, E, C)
            1,
            top_k_indices.expand(-1, -1, -1, C),  # (B, K, E, C)
        )  # (B, K, E, C)
        x = x.view(B, K, -1)  # (B, K, E*C)

        expert_logits = (
            x
            @ self.lora_experts_A.transpose(0, 1)  # (E*C, r)
            @ self.lora_experts_B.transpose(0, 1)  # (r, E*D)
        ) * top_k_probs.unsqueeze(-1).expand(
            -1, -1, -1, self.out_features
        ).contiguous().view(
            B, K, -1
        )  # (B, K, E*D)
        expert_logits = expert_logits.view(
            B, K, self.num_experts, self.out_features
        )

        if self.num_all_rounders:
            expert_logits = expert_logits * 4 / self.num_experts

        # logits: (B, T, E, C) -> (B, T, C)
        logits = torch.zeros(
            B,
            T,
            self.num_experts,
            self.out_features,
            dtype=expert_logits.dtype,
            device=x.device,
        )

        logits.scatter_(
            1,
            top_k_indices.expand(-1, -1, -1, self.out_features),
            expert_logits,
        )
        logits = logits.sum(dim=2)  # sum along the experts axis -> (B, T, D)

        return logits

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)

            if self.num_all_rounders:
                result += (
                    self.lora_dropout(x)
                    @ self.lora_all_rounder_A.transpose(0, 1)
                    @ self.lora_all_rounder_B.transpose(0, 1)
                ) * self.scaling

            if self.num_experts:
                result += self.moe_forward(self.lora_dropout(x)) * self.scaling

            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
