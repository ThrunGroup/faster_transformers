from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
)

from experiments.models.gpt2_mare.controller import apply_mare_attention
from experiments.models.gpt2_peft_mare.controller import (
    apply_mare_lora,
    freeze_model,
)
from experiments.models.gpt2_gated import apply_gated_attention
from experiments.models.gpt2_mare_upcycling.controller import (
    apply_mare_upcycling,
)

# from .gpt2_gated_no_scatter import apply_gated_attention
from utils.constants import GPT2, GPT2_MEDIUM


def build_model(
    model_name: str,
    use_mare_lora: bool = False,
    use_mare_attention: bool = False,
    use_mare_upcycling: bool = False,
    use_lora: bool = False,
    capacity_factor: int = 0,
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

    if use_mare_upcycling:
        apply_mare_upcycling(model, config, num_generalists, num_experts, capacity_factor)

    if use_mare_lora:
        apply_mare_lora(model, config, num_generalists, num_experts)
        freeze_model(model)

    return model


if __name__ == "__main__":
    model_name = GPT2
    base = build_model(model_name)
    mare_upcycling = build_model(
        model_name,
        use_mare_upcycling=True,
        num_experts=4,
        num_generalists=2,
    )

    for a, b in zip(base.transformer.h, mare_upcycling.transformer.h):
        mlp = b.mlp
        print("original")
        print(a.mlp.c_fc.weight.shape)
        print(a.mlp.c_fc.bias.shape)
        print(a.mlp.c_proj.weight.shape)
        print(a.mlp.c_proj.bias.shape)

        print("fc")
        print(mlp.c_fc_ar.weight.shape)
        print(mlp.c_fc_ar.bias.shape)
        print(mlp.c_fc_ex.weight.shape)
        print(mlp.c_fc_ex.bias.shape)
        print("proj")
        print(mlp.c_proj_ar.weight.shape)
        print(mlp.c_proj_ar.bias.shape)
        print(mlp.c_proj_ex.weight.shape)
        print(mlp.c_proj_ex.bias.shape)
        break
