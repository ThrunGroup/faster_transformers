from transformers import TrainerCallback, Trainer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import PeftModel
from datasets import Dataset
from transformers.utils import is_sagemaker_mp_enabled, is_sagemaker_dp_enabled
from typing import Any, Dict, Union, Optional, Tuple
from torch.nn import MSELoss

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy

from transformers.models.mistral.modeling_mistral import (
    MistralMLP,
    MistralAttention,
    MistralModel,
    MistralDecoderLayer,
    MistralConfig,
    MISTRAL_ATTENTION_CLASSES,
    MistralRMSNorm,
    MistralForCausalLM,
)
from experiments.models.sparse_mistral.svd_router import (
    low_rank_approximation,
    SparsePredictor,
)
from utils.utils import (
    print_size_of_model,
    is_running_deepspeed,
    is_mainprocess,
    get_datetime,
    ds_print,
)


class SparseSFTTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.regularization_coefficient = kwargs.pop("regularization_coefficient", 10)
        self.use_sparse_regularization = kwargs.pop("use_sparse_regularization", False)
        self.use_spm_loss = False
        self.freeze_original_weights = False
        self.regularization_type = kwargs.pop(
            "regularization_type", "L1 positive activation"
        )
        assert self.regularization_type in [
            "L2 activation",
            "L1 positive activation",
        ], f"Invalid regularization type: {self.regularization_type}"
        self.sparse_layers = []
        self.sparse_decoder_layers = []
        super(SparseSFTTTrainer, self).__init__(*args, **kwargs)

    def initialize_sparse_silu_layers(self, model):
        self.sparse_layers = [
            m for m in model.modules() if isinstance(m, MistralSparseSiluMLP)
        ]

    def initialize_sparse_decoder_layers(self, model):
        self.sparse_decoder_layers = [
            m for m in model.modules() if isinstance(m, SparseMistralDecoderLayer)
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
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if not self.freeze_original_weights:
            if loss is not None:
                self.accelerator.backward(loss, retain_graph=False)

        if self.use_sparse_regularization:
            regularization_loss = self.compute_regularization(model)
            if self.args.n_gpu > 1:
                regularization_loss = regularization_loss.mean()
            if regularization_loss is not None:
                self.accelerator.backward(regularization_loss, retain_graph=True)
            loss += regularization_loss

        if self.use_spm_loss:
            spm_loss = self.compute_spm_loss(model)
            if self.args.n_gpu > 1:
                spm_loss = spm_loss.mean()
            if spm_loss is not None:
                self.accelerator.backward(spm_loss, retain_graph=False)
            loss += spm_loss

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

    def compute_spm_loss(self, model):
        loss = 0
        if len(self.sparse_decoder_layers) == 0:
            self.initialize_sparse_decoder_layers(model)
        for module in self.sparse_decoder_layers:
            if module.distill_loss != None:
                loss += module.distill_loss
        if self.state.global_step % 20 == 0 and loss != 0:
            print("Sparse Predictor Distillation loss: ", loss.item())
        return loss

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     loss = super().compute_loss(model, inputs, return_outputs)
    #
    #     if is_sagemaker_mp_enabled():
    #         import smdistributed.modelparallel.torch as smp
    #         @smp.step()
    #         def smp_forward_backward(model, inputs, gradient_accumulation_steps=1):
    #             outputs = model(**inputs)
    #             loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    #             loss /= gradient_accumulation_steps
    #             model.backward(loss)
    #             return loss
    #
    #         loss_mb = smp_forward_backward(
    #             model, inputs, self.args.gradient_accumulation_steps
    #         )
    #         if self.use_sparse_regularization:
    #             return loss_mb.reduce_mean().detach().to(
    #                 self.args.device
    #             ) + self.regularization_coefficient * self.compute_regularization(model)
    #         else:
    #             return loss_mb.reduce_mean().detach().to(self)
    #
    #     if return_outputs:
    #         classification_loss, outputs = loss
    #     else:
    #         classification_loss = loss
    #
    #     loss = classification_loss
    #     if self.use_sparse_regularization:
    #         regularization_loss = self.compute_regularization(model)
    #         loss += self.regularization_coefficient * regularization_loss
    #
    #     return (loss, outputs) if return_outputs else loss


class SparseTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.regularization_coefficient = kwargs.pop("regularization_coefficient", 10)
        self.use_sparse_regularization = kwargs.pop("use_sparse_regularization", False)
        self.use_spm_loss = False
        self.freeze_original_weights = False
        self.regularization_type = kwargs.pop(
            "regularization_type", "L1 positive activation"
        )
        assert self.regularization_type in [
            "L2 activation",
            "L1 positive activation",
        ], f"Invalid regularization type: {self.regularization_type}"
        self.sparse_layers = []
        self.sparse_decoder_layers = []
        super(SparseTrainer, self).__init__(*args, **kwargs)

    def initialize_sparse_silu_layers(self, model):
        self.sparse_layers = [
            m for m in model.modules() if isinstance(m, MistralSparseSiluMLP)
        ]

    def initialize_sparse_decoder_layers(self, model):
        self.sparse_decoder_layers = [
            m for m in model.modules() if isinstance(m, SparseMistralDecoderLayer)
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
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if not self.freeze_original_weights:
            if loss is not None:
                self.accelerator.backward(loss, retain_graph=False)

        if self.use_sparse_regularization:
            regularization_loss = self.compute_regularization(model)
            if self.args.n_gpu > 1:
                regularization_loss = regularization_loss.mean()
            if regularization_loss is not None:
                self.accelerator.backward(regularization_loss, retain_graph=True)
            loss += regularization_loss

        if self.use_spm_loss:
            spm_loss = self.compute_spm_loss(model)
            if self.args.n_gpu > 1:
                spm_loss = spm_loss.mean()
            if spm_loss is not None:
                self.accelerator.backward(spm_loss, retain_graph=False)
            loss += spm_loss

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

    def compute_spm_loss(self, model):
        loss = 0
        if len(self.sparse_decoder_layers) == 0:
            self.initialize_sparse_decoder_layers(model)
        for module in self.sparse_decoder_layers:
            if module.distill_loss != None:
                loss += module.distill_loss
        if self.state.global_step % 20 == 0 and loss != 0:
            print("Sparse Predictor Distillation loss: ", loss.item())
        return loss


class SparseSiLU(nn.SiLU):
    def __init__(self, threshold):
        super(SparseSiLU, self).__init__()
        self.threshold = threshold
        self.m = nn.Threshold(self.threshold, 0)

    def set_new_threshold(self, threshold):
        self.threshold = threshold
        self.m = nn.Threshold(threshold, 0)

    def forward(self, x):
        act = super(SparseSiLU, self).forward(x)
        return self.m(act) - self.m(-act)


class MistralSparseSiluMLP(MistralMLP):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.swish_outputs = None
        self.relu = nn.ReLU()

        self.kill_sparse_swish_outputs = False
        self.dead_percentage = 0
        self.is_stats = False
        self.visit_counts = 0

        # Hyperparameters to tune
        self.dead_threshold = kwargs.pop("dead_threshold", 0)
        self.use_sparse_regularization = kwargs.pop("use_sparse_regularization", True)
        self.regularization_type = kwargs.pop(
            "regularization_type", "L1 regularization"
        )
        self.regularization_threshold = kwargs.pop("regularization_threshold", 0.5)
        self.use_relu = kwargs.pop("use_relu", False)
        self.activation_norm = None

        # Activation Histograms
        self.is_collect_histogram = False
        num_bins = 1000
        self.histogram_bins = torch.linspace(-1, 1, num_bins - 2)
        self.histogram_bins = torch.cat(
            [torch.tensor([-torch.inf]), self.histogram_bins, torch.tensor([torch.inf])]
        )
        self.pre_act_hist_counts = torch.zeros(num_bins - 1)
        self.post_act_hist_counts = torch.zeros(num_bins - 1)
        self.t = 0
        self.agg_sparsity = 0

        # Sparse activation function
        self.sparse_act_fn = SparseSiLU(threshold=self.dead_threshold)

    def activate_stats(self, is_collect_histogram: bool = True):
        self.is_stats = True
        self.dead_percentage = 0
        self.visit_counts = 0
        self.is_collect_histogram = is_collect_histogram
        self.histogram_counts = torch.zeros(2000)  # .to(self.down_proj.weight.device)

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

    def forward(
        self,
        x,
        sp_mask: torch.tensor = None,
    ):
        """
        If kill_sparse_swish_outputs is set to False, this layer functions exactly like a normal MLP layer.
        """
        if sp_mask != None:  # When sparse mask is given
            return self.down_proj(
                self.sparse_act_fn(self.gate_proj(x) * sp_mask) * self.up_proj(x)
            )  # Todo: This doesn't accelerate runtime (instead slowing down)

        elif self.use_relu:
            post_act = self.relu(self.gate_proj(x))

            if self.is_stats:
                dead_neurons = post_act == 0
                dead_percentage = dead_neurons.float().mean()
                agg_sparsity = dead_neurons.all(dim=0).float().mean()

                self.dead_percentage = (
                    self.dead_percentage * self.visit_counts + dead_percentage
                ) / (self.visit_counts + 1)
                self.agg_sparsity = (
                    self.agg_sparsity * self.visit_counts + agg_sparsity
                ) / (self.visit_counts + 1)
                self.visit_counts += 1

            return self.down_proj(post_act * self.up_proj(x))

        else:
            pre_act = self.gate_proj(x)
            post_act = self.act_fn(pre_act)
            if self.kill_sparse_swish_outputs:
                dead_neurons = post_act.abs() <= self.dead_threshold

                dead_percentage = dead_neurons.float().mean()
                agg_sparsity = dead_neurons.all(dim=0).float().mean()

                if self.is_stats:
                    self.dead_percentage = (
                        self.dead_percentage * self.visit_counts + dead_percentage
                    ) / (self.visit_counts + 1)
                    self.agg_sparsity = (
                        self.agg_sparsity * self.visit_counts + agg_sparsity
                    ) / (self.visit_counts + 1)
                    self.visit_counts += 1

                    # print(self.agg_sparsity)

                    # Collect histogram stats
                    if (
                        self.is_collect_histogram
                        and pre_act.eq(0).float().mean() < 0.99
                    ):  # Padded dataset
                        self.collect_stats(pre_act, post_act)

                post_act[dead_neurons] = 0

            out = self.down_proj(post_act * self.up_proj(x))
            if self.use_sparse_regularization:
                if self.regularization_type == "L1 regularization":
                    self.activation_norm = torch.abs(post_act)[
                        post_act < self.regularization_threshold
                    ].mean()
                elif self.regularization_type == "L2 regularization":
                    self.activation_norm = torch.sqrt(
                        torch.square(post_act)[post_act < self.regularization_threshold]
                    ).mean()

            return out


class SparseMistralDecoderLayer(MistralDecoderLayer):
    def __init__(
        self,
        config: MistralConfig,
        layer_idx: int,
        decoder_layer: MistralDecoderLayer,
        init_svd: bool = True,
        *args,
        **kwargs,
    ):
        assert isinstance(
            decoder_layer.mlp, MistralSparseSiluMLP
        ), f"{type(decoder_layer.mlp)} should MistralSparseSiluMLP."

        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.init_svd = init_svd
        self.self_attn = decoder_layer.self_attn

        self.mlp = decoder_layer.mlp
        self.input_layernorm = decoder_layer.input_layernorm
        self.post_attention_layernorm = decoder_layer.post_attention_layernorm

        # Sparse predictor for mlp (initialized with SVD decomposed matrix)
        self.low_rank = kwargs.pop("low_rank", 64)
        self.sparse_act_func = decoder_layer.mlp.sparse_act_fn

        print(
            f"Setting {layer_idx}th mlp layer's sparse predictor... svd init: {init_svd}"
        )
        self.sp_mlp = low_rank_approximation(
            decoder_layer.mlp.gate_proj,
            act_func=self.sparse_act_func,
            init_svd=init_svd,
        )
        self.use_async = kwargs.pop("use_async", False)
        self.use_sparse_predictor = False
        self.distill_loss = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        print("hidden_states shape: ", hidden_states.shape)
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states
        sp_mask = None

        if self.use_async:
            sp_mask = self.sp_mlp(hidden_states)

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if not self.use_async:
            sp_mask = self.sp_mlp(hidden_states)

        # Compute distillation loss
        gating_output = self.mlp.sparse_act_fn(self.mlp.gate_proj(hidden_states))
        loss_func = MSELoss()
        self.distill_loss = loss_func(sp_mask, gating_output)

        # Convert sp mask into binary form
        sp_mask = sp_mask > 0

        if self.training:
            sp_mask = None
        # if not self.use_sparse_predictor:
        #     sp_mask = None

        hidden_states = self.mlp(hidden_states, sp_mask)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class SparseMistralConfig(MistralConfig):
    model_type = "sparse_mistral"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SparseMistralforCausalLM(MistralForCausalLM):
    config_class = SparseMistralConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if config.use_sparse_model:
            self.apply_sparse_mlp()
            if config.thresholds is not None:
                for idx, m in enumerate(self.model.layers):
                    if isinstance(m.mlp, MistralSparseSiluMLP):
                        m.mlp.dead_threshold = config.thresholds[idx]
                        m.mlp.sparse_act_fn.set_new_threshold(m.mlp.dead_threshold)
                        m.mlp.kill_sparse_swish_outputs = True
        if config.use_sparse_predictor:
            self.apply_sparse_predictor(init_svd=config.init_svd)

    def apply_sparse_mlp(self):
        apply_mistral_sparse_silu_mlp(
            self,
            config=self.config,
            use_sparse_regularization=self.config.use_sparse_regularization,
        )

    def apply_sparse_predictor(self, init_svd: bool = True):
        apply_mistral_sparse_decoder_layer(self, config=self.config, init_svd=init_svd)


class GracefulRegularizationScheduler(TrainerCallback):
    def __init__(
        self,
        num_warmup_steps=40,
        is_enabled: bool = False,
        model_name: str = "mistral",
        test_dataset: Dataset = None,
        targeted_sparsity: float = 0.5,
        keep_regularization_with_kill: bool = False,
    ):
        """Scheduler for regularizing the model first before applying the dead threshold.

        :param num_warmup_steps: number of training steps required to reach the dead threshold, defaults to 40
        :param increment_ratio: by how much to increase the dead threshold.
            For example, 0.5 means "increase the threshold by 0.5 * desired threshold
        """
        self.num_warmup_steps = num_warmup_steps
        self.is_enabled = is_enabled
        self.model_name = model_name
        self.test_dataset = test_dataset
        self.targeted_sparsity = targeted_sparsity
        self.keep_regularization_with_kill = keep_regularization_with_kill
        self.act_hist_path = (
            f"/matx/u/vxbrando/histograms/warm_up_reg_{targeted_sparsity}/act_hist.pt"
        )
        if self.is_enabled:
            print("GracefulRegularizationScheduler is enabled.")
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if not self.is_enabled:
            return

        model = kwargs["model"]
        if isinstance(model, PeftModel):
            base_model = model.get_base_model()
        else:
            base_model = model

        if state.global_step == 1:
            ds_print("Setting an initial reg threshold to 0.1")
            set_regularization_threshold(base_model, 0.1)

        # if state.global_step >= self.num_warmup_steps and state.global_step % 50 == 0:
        if state.global_step == self.num_warmup_steps:
            activate_stats(base_model)
            enable_sparse_silu(base_model)
            self.trainer.evaluate()
            save_act_hist(base_model, self.act_hist_path)
            set_sparse_threshold(base_model, self.targeted_sparsity, True)
            deactivate_stats(base_model)
            self.trainer.use_sparse_regularization = self.keep_regularization_with_kill
            # set_layer_specific_regularization(model.get_base_model())
            print_dead_neuron_stats(model.get_base_model())

        if state.global_step % 2000 == 0:
            if is_mainprocess():
                ds_print(
                    f"Saving to /scr/lukeai/{self.model_name}_{state.global_step}.pt",
                )
                torch.save(
                    model.state_dict(),
                    f"/scr/lukeai/{self.model_name}_{state.global_step}.pt",
                )


class GradualSparsificationScheduler(TrainerCallback):
    def __init__(
        self,
        num_warmup_steps=40,
        increment_ratio=0.5,
        is_enabled: bool = False,
        model_name: str = "mistral",
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
        self.model_name = model_name

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]

        if not self.is_enabled:
            if state.global_step <= 10:
                for module in model.modules():
                    if isinstance(module, MistralSparseSiluMLP):
                        module.current_dead_threshold = module.dead_threshold
            return

        current_dead_threshold = 0
        desired_dead_threshold = 0

        if is_mainprocess():
            ds_print(state.global_step)

        if state.global_step % self.step_size == 2:
            for module in model.modules():
                if isinstance(module, MistralSparseSiluMLP):
                    desired_dead_threshold = copy.deepcopy(module.dead_threshold)
                    current_dead_threshold = module.current_dead_threshold
                    current_dead_threshold += (
                        self.increment_ratio * desired_dead_threshold
                    )
                    module.current_dead_threshold = min(
                        desired_dead_threshold, current_dead_threshold
                    )

            if is_running_deepspeed and is_mainprocess():
                ds_print(
                    state.global_step,
                    current_dead_threshold,
                    desired_dead_threshold,
                )

        if state.global_step % 2000 == 0:
            if is_running_deepspeed and is_mainprocess():
                ds_print(
                    f"Saving to /matx/u/lukeai/{self.model_name}_{state.global_step - 2}.pt",
                )
                torch.save(
                    model.state_dict(),
                    f"/matx/u/lukeai/{self.model_name}_{state.global_step - 2}.pt",
                )


def get_sparse_mistral_config(
    config: MistralConfig,
    use_sparse_model=False,
    use_sparse_predictor=False,
    use_sparse_regularization=False,
    thresholds=None,
):
    new_config = SparseMistralConfig()
    new_config.__dict__.update(config.__dict__)
    config = new_config
    config.use_sparse_model = use_sparse_model
    config.use_sparse_predictor = use_sparse_predictor
    config.use_sparse_regularization = use_sparse_regularization
    config.thresholds = thresholds

    return config


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


def apply_mistral_sparse_decoder_layer(
    model,
    config,
    init_svd: bool = True,
):
    assert isinstance(model.model, MistralModel), "model.model must be a MistralModel."
    new_layers = []
    for layer_idx, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, MistralSparseSiluMLP):
            new_layers.append(
                SparseMistralDecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                    decoder_layer=layer,
                    init_svd=init_svd,
                )
            )
            print(f"{layer_idx}th mlp layer activation: {layer.mlp.sparse_act_fn}")
        else:
            new_layers.append(layer)
    model.model.layers = nn.ModuleList(new_layers)


def enable_sparse_predictor(
    model,
):
    for layer_idx, layer in enumerate(model.model.layers):
        if isinstance(layer, MistralDecoderLayer):
            layer.use_sparse_predictor = True


def disable_sparse_predictor(
    model,
):
    for layer_idx, layer in enumerate(model.model.layers):
        if isinstance(layer, MistralDecoderLayer):
            layer.use_sparse_predictor = False


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


def print_dead_neuron_stats(model):
    total_sparsity = 0
    counts = 0
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, MistralSparseSiluMLP):
            dead_percentage = layer.mlp.dead_percentage * 100
            agg_sparsity = layer.mlp.agg_sparsity * 100
            print(f"layer {i} sparsity: {dead_percentage:.3f}%")
            print(f"layer {i} agg sparsity: {agg_sparsity:.3f}%")
            total_sparsity += dead_percentage
            counts += 1

    print(f"Total sparsity: {total_sparsity/counts: .3f}%")
    return total_sparsity / counts


def get_sparse_layers(model: MistralModel):
    sparse_layers = [
        m.mlp for m in model.layers() if isinstance(m.mlp, MistralSparseSiluMLP)
    ]
    return sparse_layers


def get_threshold(
    bin_edges: torch.tensor, histogram_counts: torch.tensor, sparsity_level: float
):  # Only for L1 Regularization
    assert (
        len(bin_edges.shape) == len(histogram_counts.shape) == 1
    ), "bin_edges and histogram are expected to be 1-dimensional."
    histogram_counts /= histogram_counts.sum()
    threshold_idx = torch.searchsorted(
        histogram_counts.cumsum(0), sparsity_level, side="right"
    )

    return bin_edges[threshold_idx]


def set_regularization_threshold(model, threshold: float = 0.1):
    for i, layer in enumerate(model.model.layers):
        if (
            isinstance(layer.mlp, MistralSparseSiluMLP) and layer.mlp.is_stats
        ):  # Can set the threshold only the relevant statistics is collected.
            layer.mlp.regularization_threshold = threshold  # TODO: find better param


def set_sparse_threshold(model, sparsity_level: float, use_relu: bool = False):
    for i, layer in enumerate(model.model.layers):
        if (
            isinstance(layer.mlp, MistralSparseSiluMLP) and layer.mlp.is_stats
        ):  # Can set the threshold only the relevant statistics is collected.
            if use_relu:
                layer.mlp.sparse_act_fn = nn.ReLU()
                layer.mlp.use_relu = True
            else:
                layer.mlp.dead_threshold = get_threshold(
                    layer.mlp.histogram_bins,
                    layer.mlp.post_act_hist_counts,
                    sparsity_level,
                )
                layer.mlp.sparse_act_fn.set_new_threshold(layer.mlp.dead_threshold)
                layer.mlp.regularization_threshold = (
                    layer.mlp.dead_threshold * 1.2
                )  # TODO: find better param


def plot_histogram(
    bin_edges, histogram_counts: torch.tensor, title: str = "Activation Distribution"
):
    plt.bar(
        bin_edges[:-1], histogram_counts, width=np.diff(bin_edges), edgecolor="black"
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
                layer.mlp.histogram_bins, layer.mlp.pre_act_hist_counts, plot_title
            )

            plot_title = f"Layer: {i} Post-Activation Absolute Distribution"
            plot_histogram(
                layer.mlp.histogram_bins, layer.mlp.post_act_hist_counts, plot_title
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


def enable_last_k_modules(model, start_module_idx: int):
    assert 32 > start_module_idx >= 0
    new_modules = []
    new_idx = 0
    for idx in range(start_module_idx, len(model.model.original_layers)):
        module = model.model.original_layers[idx]
        module.layer_idx = new_idx
        module.self_attn.layer_idx = new_idx
        new_modules.append(module)
        new_idx += 1
        print(module.layer_idx)

    model.model.layers = nn.ModuleList(new_modules)


def enable_first_k_modules(model, end_module_idx: int):
    assert 32 > end_module_idx >= 0
    new_modules = []
    new_idx = 0
    for idx in range(0, end_module_idx + 1):
        module = model.model.original_layers[idx]
        module.layer_idx = new_idx
        module.self_attn.layer_idx = new_idx
        new_modules.append(module)
        new_idx += 1
        print(module.layer_idx)

    model.model.layers = nn.ModuleList(new_modules)
