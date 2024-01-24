from transformers import TrainerCallback, Trainer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import (
    Any,
    Dict,
    Union,
)

from transformers.models.mistral.modeling_mistral import (
    MistralMLP,
    MistralAttention,
)


class SparseShared:
    def __init__(self, use_sparse_regularization=False):
        self.use_sparse_regularization = use_sparse_regularization

    def initialize_sparse_silu_layers(self, model):
        self.sparse_silu_layers = [
            m for m in model.modules() if isinstance(m, MistralSparseSiluMLP)
        ]

    def compute_regularization(self, model):
        """
        Compute a sparse regularization loss for SiLU
        """
        loss = 0
        num_layers = 32

        for module in model.modules():
            if isinstance(module, MistralSparseSiluMLP):
                if module.swish_outputs is not None:
                    swish_outputs = module.swish_outputs
                    negative_outputs = F.relu(-swish_outputs).mean()
                    loss += negative_outputs

        loss /= num_layers

        if self.state.global_step % 20 == 0 and loss != 0:
            print("Negative relularizer loss: ", loss.item())
        return loss

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Override the huggingface's training_step function to add a regularization term.
        A regularization term is computed with intermediate values, which are freed after "backward()."
        You need to set `retain_graph=True` inside `backward` function to keep the values.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = (
                loss.mean()
            )  # mean() to average on multi-gpu parallel training

        # important for a regularization loss
        self.accelerator.backward(
            loss, retain_graph=self.use_sparse_regularization
        )

        return loss.detach() / self.args.gradient_accumulation_steps


class SparseSiLUTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.regularization_coefficient = 1
        self.use_sparse_regularization = kwargs.pop(
            "use_sparse_regularization", False
        )
        super(SparseSiLUTrainer, self).__init__(*args, **kwargs)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Override the huggingface's training_step function to add a regularization term.
        A regularization term is computed with intermediate values, which are freed after "backward()."
        You need to set `retain_graph=True` inside `backward` function to keep the values.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = (
                loss.mean()
            )  # mean() to average on multi-gpu parallel training

        # important for a regularization loss
        self.accelerator.backward(
            loss, retain_graph=self.use_sparse_regularization
        )

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_regularization(self, model):
        """
        Compute a sparse regularization loss for SiLU
        """
        loss = 0
        num_layers = 32

        for module in model.modules():
            if isinstance(module, MistralSparseSiluMLP):
                if module.swish_outputs is not None:
                    swish_outputs = module.swish_outputs
                    negative_outputs = F.relu(-swish_outputs).mean()
                    loss += negative_outputs

        loss /= num_layers

        if self.state.global_step % 20 == 0 and loss != 0:
            print("Negative relularizer loss: ", loss.item())
        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = super().compute_loss(model, inputs, return_outputs)

        if return_outputs:
            classification_loss, outputs = loss
        else:
            classification_loss = loss

        loss = classification_loss
        if self.use_sparse_regularization:
            regularization_loss = self.compute_regularization(model)
            loss += self.regularization_coefficient * regularization_loss

        return (loss, outputs) if return_outputs else loss


class MistralDistributionCheck(MistralMLP):
    def __init__(self, config):
        super().__init__(config)
        self.partitions = torch.range(-5, 5, step=500)
        self.counts = torch.zeros_like(self.counts)

    # def forward(self, x):


# class MistralSparseAttention(MistralAttention):
#     def __init__(self, )


class MistralSparseSiluMLP(MistralMLP):
    def __init__(self, config):
        super().__init__(config)
        self.swish_outputs = None
        self.relu = nn.ReLU()

        self.kill_sparse_swish_outputs = False
        self.dead_percentage = None
        self.dead_threshold = 0.1

    def forward(self, x):
        """
        If kill_sparse_swish_outputs is set to False, this layer functions exactly like a normal MLP layer.
        """
        swish_outputs = self.act_fn(self.gate_proj(x))

        if self.kill_sparse_swish_outputs:
            dead_neurons = swish_outputs.abs() <= self.dead_threshold
            dead_percentage = dead_neurons.float().mean()

            if self.dead_percentage:
                # aggregate mean @jay: Have to fix this
                dead_percentage = (self.dead_percentage + dead_percentage) / 2
            self.dead_percentage = dead_percentage
            swish_outputs[dead_neurons] = 0

        out = self.down_proj(swish_outputs * self.up_proj(x))
        self.swish_outputs = swish_outputs
        return out


def apply_mistral_sparse_silu_mlp(model, config):
    for layer in model.model.layers:
        original_mlp = layer.mlp
        new_mlp = MistralSparseSiluMLP(config)
        new_mlp.gate_proj = original_mlp.gate_proj
        new_mlp.up_proj = original_mlp.up_proj
        new_mlp.down_proj = original_mlp.down_proj
        layer.mlp = new_mlp


def enable_sparse_silu(model):
    print("Enabling SparseSilu")
    for module in model.modules():
        if isinstance(module, MistralSparseSiluMLP):
            module.kill_sparse_swish_outputs = True


def print_dead_neuron_stats(model):
    for i, layer in enumerate(model.model.layers):
        dead_percentage = layer.mlp.dead_percentage * 100
        print(f"{i} : {dead_percentage:.3f}%")
