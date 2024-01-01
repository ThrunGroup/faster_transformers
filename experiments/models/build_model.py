from transformers import (
    GPT2Config,
    T5Config,
    MistralConfig,
    GPT2LMHeadModel,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
    T5ForSequenceClassification,
    MistralForSequenceClassification,
)
from torch import nn


from experiments.models.gpt2_mare.controller import apply_mare_attention
from experiments.models.gpt2_peft_mare.controller import (
    apply_mare_lora,
    freeze_model,
)
from experiments.models.gpt2_gated_attention.gpt2_gated import (
    apply_gated_attention,
)
from experiments.models.gpt2_mare_upcycling.controller import (
    apply_mare_upcycling_gpt2,
)
from experiments.models.t5_mare_upcycling.controller import (
    apply_mare_upcycling_t5,
)
from experiments.models.t5_sparity_check.t5_sparsity_check_new import (
    apply_sparsity_check_MLP,
)

from utils.constants import (
    # models
    GPT2,
    T5,
    MISTRAL_7B,
    # datasets
    BILLSUM,
    SQUAD,
    SAMSUM,
    GLUE,
    SUPERGLUE,
)


def get_model(model_type, dataset_type):
    # get config
    if GPT2 in model_type:
        config = GPT2Config
    elif model_type == MISTRAL_7B:
        config = MistralConfig
    else:
        config = T5Config

    config = config.from_pretrained(model_type)

    # get model
    if "mistral" in model_type:
        model = MistralForSequenceClassification
    elif "t5" in model_type:
        model = T5ForConditionalGeneration
        model = T5ForSequenceClassification
    elif dataset_type in [BILLSUM, SAMSUM]:
        model = AutoModelForSeq2SeqLM
        if GPT2 in model_type:
            model = GPT2LMHeadModel
    elif dataset_type in [SQUAD]:
        model = AutoModelForQuestionAnswering
    else:
        # GLUE
        model = AutoModelForSequenceClassification
    # model_type = "lukeleeai/t5-base_cola_dense_epochs5"
    model = model.from_pretrained(model_type, config=config)
    print(model)
    return config, model


def build_model(
    model_type: str,
    dataset_type: str,
    use_mare_lora: bool = False,
    use_mare_attention: bool = False,
    use_mare_upcycling: bool = False,
    use_mare_mlp: bool = False,
    capacity_factor: int = 0,
    random_token_selection: bool = False,
    num_generalists: int = 0,
    num_experts: int = 0,
    expert_size: int = 0,
    apply_to_half_layers: bool = False,
):
    config, model = get_model(model_type, dataset_type)

    if use_mare_attention:
        function_to_apply = (
            apply_mare_attention if num_generalists else apply_gated_attention
        )
        function_to_apply(
            model, config, random_token_selection, num_generalists
        )

    if use_mare_upcycling:
        function_to_apply = (
            apply_mare_upcycling_gpt2
            if GPT2 in model_type
            else apply_mare_upcycling_t5
        )
        function_to_apply(
            model,
            config,
            num_generalists,
            num_experts,
            capacity_factor,
            apply_to_half_layers,
        )

    if use_mare_lora:
        apply_mare_lora(model, config, num_generalists, num_experts)
        freeze_model(model)

    if use_mare_mlp:
        apply_sparsity_check_MLP(
            model,
            config,
            num_generalists=num_generalists,
            num_experts=num_experts,
            expert_size=expert_size,
        )

    # if dataset_type in GLUE + SUPERGLUE:
    #     if "t5" in model_type:
    #         original_out_proj = model.classification_head.out_proj
    #         new_out_proj = nn.Linear(config.d_model, 4)
    #         new_out_proj.weight.data = original_out_proj.weight.data.repeat(
    #             4, 1
    #         )
    #         new_out_proj.bias.data = original_out_proj.bias.data.repeat(4)
    #         model.classification_head.out_proj = nn.Sequential(
    #             new_out_proj, nn.Sigmoid()
    #         )
    #     elif "mistral" in model_type:
    #         model.score = nn.Sequential(
    #             nn.Linear(config.hidden_size, 4, bias=False), nn.Sigmoid()
    #         )

    return model


if __name__ == "__main__":
    model_name = T5
    model_name = "lukeleeai/t5-base_baseline"
    dataset_type = BILLSUM

    # base = build_model(model_name, dataset_type)

    # base = build_model(model_name)
    mare_upcycling = build_model(
        model_name,
        dataset_type,
        use_mare_upcycling=True,
        num_experts=8,
        num_generalists=1,
        capacity_factor=2,
    )

    print(mare_upcycling)
