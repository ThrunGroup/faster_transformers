from transformers import TrainerCallback, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import (
    Any,
    Dict,
    Union,
)

from transformers.models.mistral.modeling_mistral import MistralMLP


class ReLUTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(ReLUTrainer, self).__init__(*args, **kwargs)
        self.steps = 0
        self.relu_loss = torch.nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.alpha = 1
        self.temperature = 2.0

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = (
                loss.mean()
            )  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss, retain_graph=True)

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_penalty(self, model):
        loss = 0
        num_layers = 32

        for module in model.modules():
            if isinstance(module, MistralMLPDistillation):
                if module.swish_outputs is not None:
                    swish_outputs = module.swish_outputs
                    negative_outputs = F.relu(-swish_outputs).mean()
                    loss += negative_outputs

        loss /= num_layers

        if self.state.global_step % 20 == 0 and loss != 0:
            print("Negative relularizer loss: ", loss.item())
        return loss

    def compute_penalty_(self, model):
        loss = 0
        num_layers = 32

        for module in model.modules():
            if isinstance(module, MistralMLPDistillation):
                if (
                    module.swish_outputs is not None
                    and module.relu_outputs is not None
                ):
                    swish_outputs = (
                        module.swish_outputs.detach() / self.temperature
                    )
                    relu_outputs = (
                        module.relu_outputs.detach() / self.temperature
                    )
                    swish_outputs = F.softmax(swish_outputs, dim=1)
                    relu_outputs = F.log_softmax(relu_outputs, dim=1)
                    div_loss = nn.KLDivLoss(reduction="batchmean")(
                        relu_outputs, swish_outputs
                    )
                    loss += div_loss

        # loss /= num_layers

        if self.state.global_step % 20 == 0 and loss != 0:
            print("Distillation loss: ", loss.item())

        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        # get penalty term
        distillation_loss = self.compute_penalty(model)
        superclass_loss = super().compute_loss(model, inputs, return_outputs)

        if return_outputs:
            classification_loss, outputs = superclass_loss
        else:
            classification_loss = superclass_loss

        # Combine losses
        loss = classification_loss + self.alpha * distillation_loss

        return (loss, outputs) if return_outputs else loss


class ReluficationRateScheduler(TrainerCallback):
    def __init__(self, step_size=20, alpha_increment=0.05):
        self.step_size = step_size
        self.alpha_increment = alpha_increment

    # def on_step_end(self, args, state, control, **kwargs):
    #     model = kwargs["model"]
    #     dead_threshold = 0

    #     if state.global_step % self.step_size == 0:
    #         for module in model.modules():
    #             if isinstance(module, MistralMLPStats):
    #                 module.increase_dead_threshold(by=self.alpha_increment)
    #                 dead_threshold = module.dead_threshold

    #         print(state.global_step, dead_threshold)

    # def on_step_end(self, args, state, control, **kwargs):
    #     model = kwargs["model"]
    #     new_alpha = 0

    #     if state.global_step % self.step_size == 0:
    #         for module in model.modules():
    #             if isinstance(module, SwishToReLU):
    #                 module.increase_alpha(by=self.alpha_increment)
    #                 new_alpha = module.alpha

    #         print(state.global_step, new_alpha)


class SwishToReLU(nn.Module):
    def __init__(self):
        super(SwishToReLU, self).__init__()
        self.alpha = 0.0

    def increase_alpha(self, by=0.0):
        self.alpha = min(1.0, self.alpha + by)

    def forward(self, x):
        # (Swish -> ReLU) for negative inputs and SiLU for positive inputs
        silu = F.silu(x)
        relu_silu = (1 - self.alpha) * silu
        return torch.where(x < 0, relu_silu, silu)


class MistralMLPStats(MistralMLP):
    def __init__(self, config, max_dead_threshold=0.1):
        super().__init__(config)
        self.dead_percentage = None
        self.max_dead_threshold = max_dead_threshold
        self.dead_threshold = 2.0

    def increase_dead_threshold(self, by=0.02):
        self.dead_threshold = min(
            self.max_dead_threshold, self.dead_threshold + by
        )

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


class MistralMLPDistillation(MistralMLP):
    def __init__(self, config):
        super().__init__(config)
        self.swish_outputs = None
        self.relu_outputs = None
        self.relu = nn.ReLU()

        self.collect_stats = True
        self.dead_percentage = None
        self.dead_threshold = 0.2

    def forward(self, x):
        self.swish_outputs = self.act_fn(self.gate_proj(x))

        if self.collect_stats:
            dead_neurons = self.swish_outputs.abs() <= self.dead_threshold
            dead_percentage = dead_neurons.float().mean()
            # self.swish_outputs[dead_neurons] = 0

            if self.dead_percentage:
                # aggregate mean
                dead_percentage = (self.dead_percentage + dead_percentage) / 2
            self.dead_percentage = dead_percentage

        return self.down_proj(self.swish_outputs * self.up_proj(x))

    # def forward(self, x):
    #     with torch.no_grad():
    #         self.swish_outputs = self.down_proj(
    #             self.act_fn(self.gate_proj(x)) * self.up_proj(x)
    #         )
    #     self.relu_outputs = self.down_proj(
    #         self.relu(self.gate_proj(x)) * self.up_proj(x)
    #     )
    #     return self.relu_outputs


def apply_mistral_mlp_relu_distillation(model, config):
    for layer in model.model.layers:
        original_mlp = layer.mlp
        new_mlp = MistralMLPDistillation(config)
        new_mlp.gate_proj = original_mlp.gate_proj
        new_mlp.up_proj = original_mlp.up_proj
        new_mlp.down_proj = original_mlp.down_proj
        layer.mlp = new_mlp


def apply_mistral_mlp_stats(model, config, max_dead_threshold=0.1):
    for layer in model.model.layers:
        original_mlp = layer.mlp
        new_mlp = MistralMLPStats(config, max_dead_threshold)
        new_mlp.gate_proj = original_mlp.gate_proj
        new_mlp.up_proj = original_mlp.up_proj
        new_mlp.down_proj = original_mlp.down_proj
        layer.mlp = new_mlp


def turn_on_collect_stats(model):
    print("Turning on collect_stats")
    for module in model.modules():
        if isinstance(module, MistralMLPDistillation):
            module.collect_stats = True


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
