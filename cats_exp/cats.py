import importlib
import json
import os
from typing import List
import torch
import torch.nn as nn
import warnings

from experiments.models.sparse_silu.ugly_utils import get_threshold
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoConfig,
    LlamaForCausalLM,
    # GemmaForCausalLM,
)

from cats_exp.ops.sparse_mm import gemv_gemv_triton, gemv_triton
from transformers import modeling_utils
from utils.constants import MISTRAL_7B
from utils.utils import _get_submodules


def model_class_to_pkg_str(model_obj):
    model_class = model_obj.__class__
    return model_class.__name__
    # raise NotImplementedError(f"{model_obj.__class__} is not implemented ")


def custom_forward_for_swiglu(self, x):
    assert "MLP" in self.__class__.__name__, "{} is not an MLP".format(
        self.__class__.__name__
    )
    warnings.warn(
        "Using custom forward for MLP. Note that only batch size <= 4 is possible."
    )
    beam_width, seq_len, _ = x.shape
    if seq_len > 1:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    threshold = self.act_fn.threshold
    return gemv_gemv_triton(
        x,
        self.act_fn(self.gate_proj(x)),
        self.up_proj.weight,
        self.wdown_t,
        threshold,
    )


def custom_forward_for_simple_mlp(self, x):
    assert "MLP" in self.__class__.__name__, "{} is not an MLP".format(
        self.__class__.__name__
    )
    warnings.warn(
        "Using custom forward for MLP. Note that only batch size <= 4 is possible."
    )
    beam_width, seq_len, _ = x.shape
    if seq_len > 1:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    threshold = self.threshold
    return gemv_triton(
        x,
        self.weight_t,
        threshold,
    )


def replace_mlp_with_gemv_gemv(
    target_module,
    target_name: str,
):
    """
    Replace the forward function of the target module (llama-like MLP module) with a custom forward function that uses gemv_gemv_triton.
    """
    if "mlp" in target_name:
        print(f"Replacing {target_name} forward function with Gemv.")
        target_module.wdown_t = target_module.down_proj.weight.t().contiguous()
        setattr(
            target_module,
            "forward",
            lambda x: custom_forward_for_swiglu(target_module, x),
        )


def replace_mlp_with_gemv(
    target_module,
    target_name: str,
):
    """
    Replace the forward function of the target module (2-layer MLP module) with a custom forward function that uses gemv_triton.
    """
    if "mlp" in target_name:
        print(f"Replacing {target_name} forward function with Gemv.")
        target_module.weight_t = target_module.wrapped_module.weight.t().contiguous()
        setattr(
            target_module,
            "forward",
            lambda x: custom_forward_for_swiglu(target_module, x),
        )


class Cats(nn.Module):
    def __init__(
        self,
        wrapped_module: nn.Module,
        threshold: float = 0.0,
        hist_num_bins: int = 5000,
        hist_min: int = -5,
        hist_max: int = 5,
        pre_apply: bool = False,
    ):
        super(Cats, self).__init__()
        self.wrapped_module = wrapped_module
        self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=False)
        self.num_bins = hist_num_bins
        self.hist_min = hist_min
        self.hist_max = hist_max
        self.histogram_bins = torch.linspace(hist_min, hist_max, hist_num_bins - 2)
        self.histogram_bins = torch.cat(
            [torch.tensor([-torch.inf]), self.histogram_bins, torch.tensor([torch.inf])]
        )
        self.hist_counts = torch.zeros(hist_num_bins - 1)
        self.abs_hist_counts = torch.zeros(hist_num_bins - 1)
        self.num_zeros_act = 0
        self.num_elems_act = 0
        self.collect_stats = True
        self.print_sparsity = True
        self.is_kernel = False
        self.pre_apply = pre_apply

    def disable_collect_stats(self):
        self.collect_stats = False

    def enable_collect_stats(self):
        self.collect_stats = True

    def enable_kernel_injection(self):
        self.is_kernel = True

    def disable_kernel_injection(self):
        self.is_kernel = False

    def set_threshold(self, threshold: float):
        if isinstance(threshold, float):
            threshold = torch.tensor(threshold)
        self.threshold = nn.Parameter(threshold.clone().detach(), requires_grad=False)

    def _update_stats(self, x):
        if torch.cuda.is_available():
            pre_activation = x.float()
            post_activation = torch.abs(pre_activation)
            self.hist_counts += torch.cat(
                (
                    (pre_activation < self.hist_min).sum().unsqueeze(0),
                    torch.histc(
                        pre_activation,
                        bins=self.num_bins - 3,
                        min=self.hist_min,
                        max=self.hist_max,
                    ),
                    (pre_activation > self.hist_max).sum().unsqueeze(0),
                )
            ).cpu()
            self.abs_hist_counts += torch.cat(
                (
                    (post_activation < self.hist_min).sum().unsqueeze(0),
                    torch.histc(
                        post_activation,
                        bins=self.num_bins - 3,
                        min=self.hist_min,
                        max=self.hist_max,
                    ),
                    (post_activation > self.hist_max).sum().unsqueeze(0),
                )
            ).cpu()
        else:
            self.hist_counts += torch.histogram(x, bins=self.histogram_bins)[0]
            self.abs_hist_counts += torch.histogram(
                torch.abs(x), bins=self.histogram_bins
            )[0]

    def _update_sparsity(self, x):
        self.num_zeros_act += torch.sum(x == 0)
        self.num_elems_act += torch.numel(x)

    def forward(self, x):
        if self.pre_apply:
            if self.collect_stats:
                self._update_stats(x)
            if not self.is_kernel:
                x[abs(x) < self.threshold] = 0
            x = self.wrapped_module
        else:
            x = self.wrapped_module(x)
            if self.collect_stats:
                self._update_stats(x)
            if not self.is_kernel:
                x[abs(x) < self.threshold] = 0

        if self.print_sparsity:
            self._update_sparsity(x)

        return x


# Function to load existing data from a JSON file
def load_data(file_path):
    try:
        with open(file_path, "r") as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        return {}  # Return an empty dictionary if the file does not exist


# Function to save the dictionary to a JSON file
def save_to_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


class CatsConfig(PretrainedConfig):
    model_type = "cats_model"

    def __init__(
        self,
        wrapped_model_config=AutoConfig.from_pretrained(MISTRAL_7B),
        wrapped_model_class_name: str = "MistralForCausalLM",
        target_modules: List[str] = ["act_fn"],
        target_sparsity: float = 0.5,
        kernel_inject_targets=["mlp"],
        is_inject_kernel: bool = False,
        pre_apply: bool = False,
        **kwargs,
    ):
        self.target_modules = target_modules
        self.target_sparsity = target_sparsity
        self.kernel_inject_targets = kernel_inject_targets
        self.wrapped_model_class_name = wrapped_model_class_name
        self.is_inject_kernel = is_inject_kernel
        self.pre_apply = pre_apply
        self.__dict__.update(wrapped_model_config.__dict__)
        super().__init__(**kwargs)


class CatsModelForCausalLM(PreTrainedModel, nn.Module):
    config_class = CatsConfig

    def __init__(
        self,
        config,
        pretrained_kwargs: dict = None,
        model: PreTrainedModel = None,
    ):
        super().__init__(config)
        transformers_module = importlib.import_module("transformers")
        self.wrapped_model_class = getattr(
            transformers_module, config.wrapped_model_class_name
        )
        if pretrained_kwargs is not None:
            self.wrapped_model = self.wrapped_model_class.from_pretrained(
                **pretrained_kwargs
            )
        else:
            if model is None:
                self.wrapped_model = self.wrapped_model_class(config)
            else:
                self.wrapped_model = model
        self.kernel_inject_targets = config.kernel_inject_targets
        self.inject_cats()

    def inject_kernel(self):
        if self.kernel_inject_targets is not None:
            for name, module in self.wrapped_model.named_modules():
                parent, target, target_name = _get_submodules(self.wrapped_model, name)
                if target_name in self.kernel_inject_targets:
                    print(f"injecting kernel to {name}")
                    if target_name in self.kernel_inject_targets:
                        if isinstance(target, nn.Linear):
                            replace_mlp_with_gemv(target, target_name)
                        elif isinstance(target, modeling_utils.Swiglu):
                            replace_mlp_with_gemv_gemv(target, target_name)
                        replace_mlp_with_gemv_gemv(target, target_name)
                if isinstance(target, Cats):
                    target.enable_kernel_injection()

    def inject_cats(self):
        for name, module in self.wrapped_model.named_modules():
            parent, target, target_name = _get_submodules(self.wrapped_model, name)
            if target_name in self.config.target_modules:
                # print(f"{name} is replaced.")

                # Replace target module with target module + CATS
                cats = Cats(wrapped_module=target, pre_apply=self.config.pre_apply)
                setattr(parent, target_name, cats)

    def enable_collect_stats(self):
        print("enable stats")
        for name, module in self.wrapped_model.named_modules():
            if isinstance(module, Cats):
                module.enable_collect_stats()
                module.print_sparsity = True

    def disable_collect_stats(self) -> None:
        print("disable stats")

        for name, module in self.wrapped_model.named_modules():
            if isinstance(module, Cats):
                module.disable_collect_stats()
                module.print_sparsity = False

    def init_stats(self) -> None:
        print("Initialize stats")

        for name, module in self.wrapped_model.named_modules():
            if isinstance(module, Cats):
                module.num_elems_act = 0
                module.num_zeros_act = 0

    def set_thresholds(self):
        print("Setting threshold")
        for name, module in self.wrapped_model.named_modules():
            if isinstance(module, Cats):
                threshold = get_threshold(
                    module.histogram_bins,
                    module.abs_hist_counts,
                    self.config.target_sparsity,
                )
                module.set_threshold(threshold)

    def get_sparsity(self):
        sparsity_dict = {}
        idx = 0
        for name, module in self.wrapped_model.named_modules():
            if isinstance(module, Cats):
                sparsity_dict[name] = float(module.num_zeros_act / module.num_elems_act)
                idx += 1
        return sparsity_dict

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.wrapped_model, name)

    def __call__(self, *args, **kwargs):
        """Override the __call__ method to use the wrapped model's forward method."""
        return self.wrapped_model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.wrapped_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.wrapped_model.generate(*args, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.wrapped_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)


def get_cats_model(
    model,
    target_modules: List[str] = ["act_fn"],
    target_sparsity: float = 0.5,
    sparsity_type: str = "Constant",
    kernel_inject_targets: List[str] = ["mlp"],
    is_inject_kernel: bool = False,
    is_share_params: bool = True,
):
    config = model.config
    wrapped_model_class_name = model_class_to_pkg_str(model)

    cats_config = CatsConfig(
        config,
        wrapped_model_class_name,
        target_modules=target_modules,
        target_sparsity=target_sparsity,
        kernel_inject_targets=kernel_inject_targets,
        is_inject_kernel=is_inject_kernel,
    )
    pretrained_kwargs = {
        "pretrained_model_name_or_path": config.name_or_path,
        "torch_dtype": model.dtype,
        "attn_implementation": config._attn_implementation,
    }
    # print(pretrained_kwargs)
    if is_share_params:
        cats_model = CatsModelForCausalLM(cats_config, model=model)
    else:
        cats_model = CatsModelForCausalLM(
            cats_config,
            pretrained_kwargs=pretrained_kwargs,
            model=None,
        )

    return cats_model
