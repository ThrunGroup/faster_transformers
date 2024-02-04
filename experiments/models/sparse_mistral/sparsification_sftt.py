from transformers import TrainerCallback, Trainer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
from transformers.utils import is_sagemaker_mp_enabled, is_sagemaker_dp_enabled

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from typing import (
    Any,
    Dict,
    Union,
)

from transformers.models.mistral.modeling_mistral import (
    MistralMLP,
    MistralForCausalLM,
    MistralModel,
)


class SparseTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.regularization_coefficient = kwargs.pop(
            "regularization_coefficient", 10
        )
        self.use_sparse_regularization = kwargs.pop(
            "use_sparse_regularization", False
        )
        self.regularization_type = kwargs.pop(
            "regularization_type", "L1 positive activation"
        )
        assert self.regularization_type in [
            "L2 activation",
            "L1 positive activation",
        ], f"Invalid regularization type: {self.regularization_type}"
        self.sparse_layers = []
        super(SparseTrainer, self).__init__(*args, **kwargs)

    def initialize_sparse_silu_layers(self, model):
        self.sparse_layers = [
            m for m in model.modules() if isinstance(m, MistralSparseSiluMLP)
        ]

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

        self.accelerator.backward(
            loss, retain_graph=self.use_sparse_regularization
        )

        if self.use_sparse_regularization:
            regularization_loss = self.compute_regularization(model)
            if self.args.n_gpu > 1:
                regularization_loss = regularization_loss.mean()
            self.accelerator.backward(regularization_loss, retain_graph=False)
            loss += regularization_loss

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_regularization(self, model):
        """
        Compute a sparse regularization loss for SiLU
        """
        loss = 0
        if len(self.sparse_layers) == 0:
            self.initialize_sparse_silu_layers(model)
        num_layers = len(self.sparse_layers)

        for module in self.sparse_layers:
            if module.activation_norm is not None:
                loss += module.activation_norm

        loss /= num_layers
        loss *= self.regularization_coefficient

        if self.state.global_step % 20 == 0 and loss != 0:
            print("Negative relularizer loss: ", loss.item())
        return loss


class SparseSFTTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.regularization_coefficient = kwargs.pop(
            "regularization_coefficient", 10
        )
        self.use_sparse_regularization = kwargs.pop(
            "use_sparse_regularization", False
        )
        self.regularization_type = kwargs.pop(
            "regularization_type", "L1 positive activation"
        )
        assert self.regularization_type in [
            "L2 activation",
            "L1 positive activation",
        ], f"Invalid regularization type: {self.regularization_type}"
        self.sparse_layers = []
        super(SparseSFTTTrainer, self).__init__(*args, **kwargs)

    def initialize_sparse_silu_layers(self, model):
        self.sparse_layers = [
            m for m in model.modules() if isinstance(m, MistralSparseSiluMLP)
        ]

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

        self.accelerator.backward(
            loss, retain_graph=self.use_sparse_regularization
        )

        if self.use_sparse_regularization:
            regularization_loss = self.compute_regularization(model)
            if self.args.n_gpu > 1:
                regularization_loss = regularization_loss.mean()
            self.accelerator.backward(regularization_loss, retain_graph=False)
            loss += regularization_loss

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_regularization(self, model):
        """
        Compute a sparse regularization loss for SiLU
        """
        loss = 0
        if len(self.sparse_layers) == 0:
            self.initialize_sparse_silu_layers(model)
        num_layers = len(self.sparse_layers)

        for module in self.sparse_layers:
            if module.activation_norm is not None:
                loss += module.activation_norm

        loss /= num_layers
        loss *= self.regularization_coefficient

        if self.state.global_step % 20 == 0 and loss != 0:
            print("Negative relularizer loss: ", loss.item())
        return loss


class SparseMistral(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        apply_mistral_sparse_silu_mlp(self, config)


class MistralSparseSiluMLP(MistralMLP):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.swish_outputs = None
        # self.relu = nn.ReLU()

        self.kill_sparse_swish_outputs = False
        self.dead_percentage = 0
        self.is_stats = False
        self.visit_counts = 0

        # Hyperparameters to tune
        self.dead_threshold = kwargs.pop("dead_threshold", 0)
        self.current_dead_threshold = 0.0
        self.use_sparse_regularization = kwargs.pop(
            "use_sparse_regularization", True
        )
        self.regularization_type = kwargs.pop(
            "regularization_type", "L1 regularization"
        )
        self.regularization_threshold = kwargs.pop(
            "regularization_threshold", 0.0
        )
        self.activation_norm = None

        # Activation Histograms
        self.is_collect_histogram = False
        num_bins = 1000
        self.histogram_bins = torch.linspace(-1, 1, num_bins - 2)
        self.histogram_bins = torch.cat(
            [
                torch.tensor([-torch.inf]),
                self.histogram_bins,
                torch.tensor([torch.inf]),
            ]
        )
        self.pre_act_hist_counts = torch.zeros(num_bins - 1)
        self.post_act_hist_counts = torch.zeros(num_bins - 1)
        self.t = 0

    def activate_stats(self, is_collect_histogram: bool = True):
        self.is_stats = True
        self.dead_percentage = 0
        self.visit_counts = 0
        self.is_collect_histogram = is_collect_histogram
        self.histogram_counts = torch.zeros(
            2000
        )  # .to(self.down_proj.weight.device)

    def deactivate_stats(self):
        self.is_stats = False

    def collect_stats(self, pre_activation, post_activation):
        start_time = time.time()
        pre_activation = pre_activation.float().cpu().detach()
        post_activation = post_activation.float().cpu().detach()
        # self.histogram_bins=self.histogram_bins.to(pre_activation.device).type(pre_activation.dtype)
        self.pre_act_hist_counts += torch.histogram(
            pre_activation, bins=self.histogram_bins
        )[0]
        self.post_act_hist_counts += torch.histogram(
            torch.abs(post_activation), bins=self.histogram_bins
        )[0]
        self.t += time.time() - start_time
        if self.visit_counts % 30 == 0:
            print(f"Time taken to collect stats: {self.t}s.")

    def forward(self, x):
        """
        If kill_sparse_swish_outputs is set to False, this layer functions exactly like a normal MLP layer.
        """
        pre_act = self.gate_proj(x)
        post_act = self.act_fn(pre_act)

        if self.use_sparse_regularization:
            if self.regularization_type == "L1 regularization":
                self.activation_norm = torch.abs(post_act)[
                    post_act < self.regularization_threshold
                ].mean()
            elif self.regularization_type == "L2 regularization":
                self.activation_norm = torch.sqrt(
                    torch.square(post_act)[
                        post_act < self.regularization_threshold
                    ]
                ).mean()

        if self.kill_sparse_swish_outputs:
            dead_neurons = post_act.abs() <= self.current_dead_threshold
            dead_percentage = dead_neurons.float().mean()

            if self.is_stats:
                self.dead_percentage = (
                    self.dead_percentage * self.visit_counts + dead_percentage
                ) / (self.visit_counts + 1)
                self.visit_counts += 1

                # Collect histogram stats
                if self.is_collect_histogram:
                    self.collect_stats(pre_act, post_act)

            post_act[dead_neurons] = 0

        out = self.down_proj(post_act * self.up_proj(x))

        return out


class GradualSparsificationScheduler(TrainerCallback):
    def __init__(
        self,
        num_warmup_steps=40,
        increment_ratio=0.5,
        is_enabled: bool = False,
    ):
        """Scheduler for gradually increasing a dead threshold until it reaches the desired threshold.

        :param num_warmup_steps: number of training steps required to reach the dead threshold, defaults to 40
        :param increment_ratio: by how much to increase the dead threshold.
            For example, 0.5 means "increase the threshold by 0.5 * desired threshold
        """
        self.num_warmup_steps = num_warmup_steps
        self.increment_ratio = increment_ratio
        self.step_size = int(num_warmup_steps * increment_ratio)
        self.is_enabled = is_enabled

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]

        if not self.is_enabled:
            if state.global_step <= 10:
                print("NO WARM UP")
                for module in model.modules():
                    if isinstance(module, MistralSparseSiluMLP):
                        module.current_dead_threshold = module.dead_threshold
            return

        current_dead_threshold = 0

        if state.global_step % self.step_size == 0:
            for module in model.modules():
                if isinstance(module, MistralSparseSiluMLP):
                    desired_dead_threshold = module.dead_threshold
                    current_dead_threshold = module.current_dead_threshold
                    current_dead_threshold += (
                        self.increment_ratio * desired_dead_threshold
                    )
                    module.current_dead_threshold = min(
                        desired_dead_threshold, current_dead_threshold
                    )

            print(state.global_step, current_dead_threshold)


def apply_mistral_sparse_silu_mlp(
    model,
    config,
    use_sparse_regularization: bool = False,
):
    # counts = 0
    for layer in model.model.layers:
        # counts += 1
        # if counts < 4:
        #     continue
        original_mlp = layer.mlp
        new_mlp = MistralSparseSiluMLP(
            config, use_sparse_regularization=use_sparse_regularization
        )
        new_mlp.gate_proj = original_mlp.gate_proj
        new_mlp.up_proj = original_mlp.up_proj
        new_mlp.down_proj = original_mlp.down_proj
        layer.mlp = new_mlp


def activate_stats(model, is_collect_histogram: bool = True):
    for layer in model.model.layers:
        if isinstance(layer.mlp, MistralSparseSiluMLP):
            layer.mlp.activate_stats(is_collect_histogram=is_collect_histogram)


def deactivate_stats(model):
    for layer in model.model.layers:
        if isinstance(layer.mlp, MistralSparseSiluMLP):
            layer.mlp.deactivate_stats()


def enable_sparse_silu(model):
    print("Enabling SparseSilu")
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, MistralSparseSiluMLP):
            layer.mlp.kill_sparse_swish_outputs = True


def enable_relu(model):
    print("Enabling ReLU")
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, MistralSparseSiluMLP):
            layer.mlp.act_fn = nn.ReLU()


def print_dead_neuron_stats(model):
    total_sparsity = 0
    counts = 0
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, MistralSparseSiluMLP):
            dead_percentage = layer.mlp.dead_percentage * 100
            print(f"{i} : {dead_percentage:.3f}%")
            total_sparsity += dead_percentage
            counts += 1

    print(f"Total sparsity: {total_sparsity/counts: .3f}%")
    return total_sparsity / counts


def get_sparse_layers(model: MistralModel):
    sparse_layers = [
        m.mlp
        for m in model.layers()
        if isinstance(m.mlp, MistralSparseSiluMLP)
    ]
    return sparse_layers


def get_threshold(
    bin_edges: torch.tensor,
    histogram_counts: torch.tensor,
    sparsity_level: float,
):  # Only for L1 Regularization
    assert (
        len(bin_edges.shape) == len(histogram_counts.shape) == 1
    ), "bin_edges and histogram are expected to be 1-dimensional."
    histogram_counts /= histogram_counts.sum()
    threshold_idx = torch.searchsorted(
        histogram_counts.cumsum(0), sparsity_level, side="right"
    )

    return bin_edges[threshold_idx]


def set_sparse_threshold(model, sparsity_level: float):
    for i, layer in enumerate(model.model.layers):
        if (
            isinstance(layer.mlp, MistralSparseSiluMLP) and layer.mlp.is_stats
        ):  # Can set the threshold only the relevant statistics is collected.
            layer.mlp.dead_threshold = get_threshold(
                layer.mlp.histogram_bins,
                layer.mlp.post_act_hist_counts,
                sparsity_level,
            )
            print(i, get_threshold)
            layer.mlp.regularization_threshold = (
                layer.mlp.dead_threshold * 1.2
            )  # TODO: find better param


def plot_histogram(
    bin_edges,
    histogram_counts: torch.tensor,
    title: str = "Activation Distribution",
):
    plt.bar(
        bin_edges[:-1],
        histogram_counts,
        width=np.diff(bin_edges),
        edgecolor="black",
    )
    plt.title(title)
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{title}.png")
    # plt.show()
    plt.clf()


def plot_act(model):
    for i, layer in enumerate(model.model.layers):
        if (
            isinstance(layer.mlp, MistralSparseSiluMLP) and layer.mlp.is_stats
        ):  # Can set the threshold only the relevant statistics is collected.
            plot_title = f"Layer: {i} Pre-Activation Distribution"
            plot_histogram(
                layer.mlp.histogram_bins,
                layer.mlp.pre_act_hist_counts,
                plot_title,
            )

            plot_title = f"Layer: {i} Post-Activation Absolute Distribution"
            plot_histogram(
                layer.mlp.histogram_bins,
                layer.mlp.post_act_hist_counts,
                plot_title,
            )


def save_act_hist(
    model, filename="/scr/jay/models/mistral/pre_finetune/cola_act_hist.pt"
):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    act_dict = {}
    for i, layer in enumerate(model.model.layers):
        if (
            isinstance(layer.mlp, MistralSparseSiluMLP) and layer.mlp.is_stats
        ):  # Can set the threshold only the relevant statistics is collected.
            act_dict[i] = (
                layer.mlp.histogram_bins,
                layer.mlp.pre_act_hist_counts,
                layer.mlp.post_act_hist_counts,
            )
    print("Saving activation histograms...\n\n\n")
    torch.save(act_dict, filename)


def load_act_hist(
    model, filename="/scr/jay/models/mistral/pre_finetune/cola_act_hist.pt"
):
    assert os.path.exists(
        filename
    ), f"{filename} does not exist when loading pre/post-activation histogram of SparseMistralSiluMLP."
    print("Loading activation histograms...\n\n\n")

    act_dict = torch.load(filename)
    for i, layer in enumerate(model.model.layers):
        if (
            isinstance(layer.mlp, MistralSparseSiluMLP) and layer.mlp.is_stats
        ):  # Can set the threshold only the relevant statistics is collected.
            (
                layer.mlp.histogram_bins,
                layer.mlp.pre_act_hist_counts,
                layer.mlp.post_act_hist_counts,
            ) = act_dict[i]
