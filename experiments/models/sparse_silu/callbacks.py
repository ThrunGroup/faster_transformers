from transformers import TrainerCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy

from experiments.models.sparse_mistral.sparse_silu import *
from experiments.models.sparse_llama.sparse_silu import *
from utils.utils import is_running_deepspeed, is_mainprocess, ds_print, get_model_type
from utils.constants import LLAMA, MISTRAL


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
        self.act_hist_path = f"/scr/lukeai/histograms/warm_up_reg_{targeted_sparsity}/act_hist.pt"
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
            disable_sparse_silu(base_model)

        if state.global_step == self.num_warmup_steps:
            activate_stats(base_model)
            enable_sparse_silu(base_model)
            self.trainer.evaluate()
            save_act_hist(base_model, self.act_hist_path)
            set_sparse_threshold(base_model, self.targeted_sparsity, False)
            deactivate_stats(base_model)
            self.trainer.use_sparse_regularization = self.keep_regularization_with_kill
            print_dead_neuron_stats(model.get_base_model())


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
        self.model_type = get_model_type(model_name)
        self.mlp_type = MistralSparseSiluMLP if self.model_type == MISTRAL else LlamaSparseSiluMLP

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]

        if not self.is_enabled:
            if state.global_step <= 10:
                for module in model.modules():
                    if isinstance(module, self.mlp_type):
                        module.current_dead_threshold = module.dead_threshold
            return

        current_dead_threshold = 0
        desired_dead_threshold = 0

        if is_mainprocess():
            ds_print(state.global_step)

        if state.global_step % self.step_size == 2:
            for module in model.modules():
                if isinstance(module, self.mlp_type):
                    desired_dead_threshold = copy.deepcopy(module.dead_threshold)
                    current_dead_threshold = module.current_dead_threshold
                    current_dead_threshold += self.increment_ratio * desired_dead_threshold
                    module.current_dead_threshold = min(desired_dead_threshold, current_dead_threshold)

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
