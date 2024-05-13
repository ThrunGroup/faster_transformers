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
    AutoConfig, AutoModelForCausalLM,
)

from utils.constants import MISTRAL_7B
from utils.utils import _get_submodules
from experiments.models.cats.cats import Cats, CatsModel, CatsConfig

def simple_exp():
    model_dir = MISTRAL_7B
    config = AutoConfig.from_pretrained(model_dir)
    cats_config = CatsConfig(config, wrapped_model_class_name="MistralForCausalLM")
    model = CatsModel(cats_config, wrapped_model_pretrained_dir=MISTRAL_7B)
    print(model)
    print(model.wrapped_model)
    print(model.config)

    CatsConfig.register_for_auto_class()
    CatsModel.register_for_auto_class("AutoModelForCausalLM")

    local_dir = "simple_exp"
    # model.save_pretrained(local_dir, variant="fp16")
    # model = AutoModelForCausalLM.from_pretrained(local_dir, trust_remote_code=True)



if __name__ == "__main__":
    simple_exp()