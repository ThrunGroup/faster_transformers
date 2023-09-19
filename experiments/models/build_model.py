from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
)
import torch

from experiments.models.gpt2_mare.controller import apply_mare_attention
from experiments.models.gpt2_peft_mare.controller import (
    apply_mare_lora,
    freeze_model,
)
from experiments.models.gpt2_gated import apply_gated_attention

# from .gpt2_gated_no_scatter import apply_gated_attention
from utils.constants import GPT2_MEDIUM


def build_model(
    model_name: str,
    use_mare_lora: bool = False,
    use_mare_attention: bool = False,
    use_lora: bool = False,
    random_token_selection: bool = False,
    num_generalists: int = 0,
    num_experts: int = 0,
):
    config = GPT2Config.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)

    if use_mare_attention:
        function_to_apply = (
            apply_mare_attention if num_generalists else apply_gated_attention
        )
        function_to_apply(
            model, config, random_token_selection, num_generalists
        )

    if use_mare_lora:
        apply_mare_lora(model, config, num_generalists, num_experts)
        freeze_model(model)

    return model


if __name__ == "__main__":
    model_name = GPT2_MEDIUM
    base = build_model(model_name)
    marelora = build_model(model_name, use_mare_lora=True)

    for a, b in zip(base.transformer.h, marelora.transformer.h):
        # print(b.mlp.c_fc.r)
        print(torch.allclose(a.mlp.c_fc.weight, b.mlp.c_fc.weight))
