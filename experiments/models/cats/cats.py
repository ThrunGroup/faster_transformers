import importlib
import json
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoConfig,
    MistralForCausalLM,
)

from utils.constants import MISTRAL_7B
from utils.utils import _get_submodules


# class Cats(nn.Module):
#     def __init__(
#         self,
#         act_class=nn.SiLU,
#         sparsity_level: float = 0.5,
#         threshold: float = None,
#     ):
#         super(Cats, self).__init__()
#         self.silu = act_class()
#         self.sparsity_level = sparsity_level
#         self.num_params = 0
#         self.num_zeros = 0
#         self.threshold = threshold
#         self.new_threshold = 0
#         self.num_samples = 0
#
#     def forward(self, x):
#         x = self.silu(x)
#         if self.threshold is not None:
#             x[abs(x) < self.threshold] = 0
#         else:
#             if self.num_samples <= 2:
#                 q = torch.quantile(
#                     torch.abs(x.to(torch.float32)), self.sparsity_level, dim=1
#                 )[:, None]
#                 self.new_threshold = (
#                     self.new_threshold * self.num_samples + torch.mean(q)
#                 ) / (self.num_samples + 1)
#
#             x[torch.abs(x) < q] = 0
#             self.num_samples += 1
#         self.num_zeros += torch.sum(x == 0)
#         self.num_params += torch.numel(x)
#         return x
class Cats(nn.Module):
    def __init__(
        self,
        wrapped_module: nn.Module,
        threshold: float = 0,
        hist_num_bins: int = 1000,
        hist_min: int = -1,
        hist_max: int = 1,
    ):
        super(Cats, self).__init__()
        self.wrapped_module = wrapped_module
        self.threshold = threshold
        self.histogram_bins = torch.linspace(hist_min, hist_max, hist_num_bins - 2)
        self.histogram_bins = torch.cat(
            [torch.tensor([-torch.inf]), self.histogram_bins, torch.tensor([torch.inf])]
        )
        self.hist_counts = torch.zeros(hist_num_bins - 1)
        self.abs_hist_counts = torch.zeros(hist_num_bins - 1)
        self.collect_stats = True

    def disable_collect_stats(self):
        self.collect_stats = False

    def enable_collect_stats(self):
        self.collect_stats = True

    def set_threshold(self, threshold: float):
        self.threshold = threshold

    def forward(self, x):
        x = self.wrapped_module(x)
        if self.collect_stats:
            self.hist_counts += torch.histogram(x, bins=self.histogram_bins)[0]
            self.abs_hist_counts += torch.histogram(
                torch.abs(x), bins=self.histogram_bins
            )[0]
        x[abs(x) < self.threshold] = 0
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


def inject_cats(model, act_class=nn.SiLU):
    thresholds = [
        0.017051145434379578,
        0.023069201037287712,
        0.049147434532642365,
        0.051153454929590225,
        0.061183542013168335,
        0.07121363282203674,
        0.0732196494936943,
        0.07923770695924759,
        0.08324974030256271,
        0.08124372363090515,
        0.089267797768116,
        0.09127381443977356,
        0.10130390524864197,
        0.0992978885769844,
        0.10732196271419525,
        0.12337010353803635,
        0.14343027770519257,
        0.16148445010185242,
        0.17953860759735107,
        0.1935807317495346,
        0.1995987892150879,
        0.2196589708328247,
        0.2196589708328247,
        0.23169508576393127,
        0.2357071191072464,
        0.23370109498500824,
        0.225677028298378,
        0.22968906164169312,
        0.225677028298378,
        0.22768303751945496,
        0.2457372099161148,
        0.2678034007549286,
    ]
    thresholds = np.zeros_like(thresholds)
    idx = 0
    for name, module in model.named_modules():
        if isinstance(module, act_class):
            if "silu" in name:
                print(module)
            print(f"{idx}-{name} is replaced!")
            if "." not in name:
                setattr(
                    model, name, Cats(act_class=act_class, threshold=thresholds[idx])
                )
            else:
                parent_name, child_name = name.rsplit(".", 1)
                parent_module = dict(model.named_modules())[parent_name]
                setattr(
                    parent_module,
                    child_name,
                    Cats(act_class=act_class, threshold=thresholds[idx]),
                )
            idx += 1


class CatsConfig(PretrainedConfig):
    def __init__(
        self,
        wrapped_model_config=AutoConfig.from_pretrained(MISTRAL_7B),
        wrapped_model_class_name: str = "MistralForCausalLM",
        target_modules: List[str] = ["silu"],
        target_sparsity: float = 0.5,
        **kwargs,
    ):
        # self.wrapped_model_config = wrapped_model_config
        self.target_modules = target_modules
        self.target_sparsity = target_sparsity
        self.wrapped_model_class_name = wrapped_model_class_name
        self.kwargs = kwargs
        self.__dict__.update(wrapped_model_config.__dict__)
        super().__init__(**kwargs)


class CatsForCausalLM(PreTrainedModel):
    config_class = CatsConfig

    def __init__(self, config, wrapped_model_pretrained_dir: str = None, **kwargs):
        super().__init__(config)
        print(self.config)
        transformers_module = importlib.import_module("transformers")
        self.wrapped_model_class = getattr(
            transformers_module, config.wrapped_model_class_name
        )
        print(self.wrapped_model_class)
        self.wrapped_model = self.wrapped_model_class(config)
        # self.wrapped_model = MistralForCausalLM(config.wrapped_model_config)
        if wrapped_model_pretrained_dir is not None:
            self.wrapped_model = self.wrapped_model_class.from_pretrained(
                wrapped_model_pretrained_dir
            )
        print(self.wrapped_model)

        print(self.config)
        self.inject_cats()

    def inject_cats(self):
        for name, module in self.wrapped_model.named_modules():
            print(name)
            parent, target, target_name = _get_submodules(self.wrapped_model, name)
            if target_name in self.config.target_modules:
                print(f"{target_name} is replaced.")
                # Replace target module with target module + CATS
                cats = Cats(wrapped_module=target)
                setattr(parent, target_name, cats)

    def enable_collect_stats(self):
        for module in self.wrapped_model.named_modules():
            if isinstance(module, Cats):
                module.enable_collect_stats()

    def disable_adapters(self) -> None:
        for module in self.wrapped_model.named_modules():
            if isinstance(module, Cats):
                module.disable_collect_stats()

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        return getattr(self.wrapped_model, name)


def simple_exp():
    model_dir = MISTRAL_7B
    config = AutoConfig.from_pretrained(model_dir)
    # a = MistralForCausalLM(config)
    # print(a)
    cats_config = CatsConfig(config, wrapped_model_class_name="MistralForCausalLM")
    model = CatsForCausalLM(cats_config, wrapped_model_pretrained_dir=None)
    print(model)


if __name__ == "__main__":
    simple_exp()
