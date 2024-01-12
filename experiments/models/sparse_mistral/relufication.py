from transformers import TrainerCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ReluficationRateScheduler(TrainerCallback):
    def __init__(self, step_size=10, alpha_increment=0.01):
        self.step_size = step_size
        self.alpha_increment = alpha_increment

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        new_alpha = 0

        if state.global_step % self.step_size == 5:
            for module in model.modules():
                if isinstance(module, SwishToReLU):
                    model.increase_alpha(by=self.alpha_increment)
                    new_alpha = model.alpha

            print("New alpha: ", new_alpha)


class SwishToReLU(nn.Module):
    def __init__(self):
        super(SwishToReLU, self).__init__()
        self.alpha = 0.0

    def increase_alpha_logarithmically(self, epoch):
        self.alpha = min(1.0, math.log10(epoch + 1))

    def increase_alpha(self, by=0.0):
        self.alpha = min(1.0, self.alpha + by)

    def forward(self, x):
        # (Swish -> ReLU) for negative inputs and SiLU for positive inputs
        silu_part = F.silu(x)
        relu_silu_part = self.alpha * F.relu(x) + (1 - self.alpha) * F.silu(x)
        return torch.where(x < 0, relu_silu_part, silu_part)


def apply_relufication(
    model,
):
    print(model)
    for layers in model.model.layers:
        layers.mlp.act_fn = SwishToReLU()
