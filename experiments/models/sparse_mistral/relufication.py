from transformers import TrainerCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers.models.mistral.modeling_mistral import MistralMLP


class ReluficationRateScheduler(TrainerCallback):
    def __init__(self, step_size=100, alpha_increment=0.3):
        self.step_size = step_size
        self.alpha_increment = alpha_increment

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        new_alpha = 0

        if state.global_step % self.step_size == 0:
            for module in model.modules():
                if isinstance(module, SwishToReLU):
                    module.increase_alpha(by=self.alpha_increment)
                    new_alpha = module.alpha

            print(state.global_step, new_alpha)


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
        silu = F.silu(x)
        relu_silu = self.alpha * F.relu(x) + (1 - self.alpha) * silu
        return torch.where(x < 0, relu_silu, silu)


class MistralMLPStats(MistralMLP):
    def __init__(self, config, dead_threshold=0.1):
        super().__init__(config)
        self.dead_percentage = None
        self.dead_threshold = dead_threshold

    def forward(self, x):
        swish = self.act_fn(self.gate_proj(x))

        dead_neurons = swish.abs() <= self.dead_threshold
        dead_percentage = dead_neurons.float().mean()
        swish[dead_neurons] = 0

        if self.dead_percentage:
            # aggregate mean
            dead_percentage = (self.dead_percentage + dead_percentage) / 2
        self.dead_percentage = dead_percentage

        return self.down_proj(swish * self.up_proj(x))


def apply_mistral_mlp_stats(model, config, dead_threshold=0.1):
    for layer in model.model.layers:
        original_mlp = layer.mlp
        new_mlp = MistralMLPStats(config, dead_threshold)
        new_mlp.gate_proj = original_mlp.gate_proj
        new_mlp.up_proj = original_mlp.up_proj
        new_mlp.down_proj = original_mlp.down_proj
        layer.mlp = new_mlp


def print_dead_neuron_stats(model):
    for i, layer in enumerate(model.model.layers):
        dead_percentage = layer.mlp.dead_percentage * 100
        print(f"{i} : {dead_percentage:.3f}%")


def apply_relufication(
    model,
):
    print(model)
    for layers in model.model.layers:
        layers.mlp.act_fn = SwishToReLU()
