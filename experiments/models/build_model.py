from transformers import (
    GPT2Config,
    T5Config,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
    T5ForSequenceClassification,
)
import torch

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

from utils.constants import (
    GPT2,
    T5,
    BILLSUM,
    SQUAD,
    SST2,
    SAMSUM,
    QNLI,
    COLA,
)


def get_model(model_type, dataset_type):
    # get config
    if GPT2 in model_type:
        config = GPT2Config
    else:
        config = T5Config
    config = config.from_pretrained(model_type)

    # get model
    if "t5" in model_type:
        model = T5ForConditionalGeneration
        if dataset_type in [QNLI, SST2, COLA]:
            model = T5ForSequenceClassification

    elif dataset_type in [BILLSUM, SAMSUM]:
        model = AutoModelForSeq2SeqLM
    elif dataset_type in [SQUAD]:
        model = AutoModelForQuestionAnswering
    else:
        # GLUE
        model = AutoModelForSequenceClassification
    model = model.from_pretrained(model_type, config=config)

    return config, model


def build_model(
    model_type: str,
    dataset_type: str,
    use_mare_lora: bool = False,
    use_mare_attention: bool = False,
    use_mare_upcycling: bool = False,
    capacity_factor: int = 0,
    random_token_selection: bool = False,
    num_generalists: int = 0,
    num_experts: int = 0,
    apply_to_half_layers: bool = False,
):
    config, model = get_model(model_type, dataset_type)

    if use_mare_attention:
        function_to_apply = (
            apply_mare_attention if num_generalists else apply_gated_attention
        )
        function_to_apply(model, config, random_token_selection, num_generalists)

    if use_mare_upcycling:
        function_to_apply = (
            apply_mare_upcycling_gpt2 if GPT2 in model_type else apply_mare_upcycling_t5
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

    return model


if __name__ == "__main__":
    model_name = T5
    dataset_type = BILLSUM

    base = build_model(model_name, dataset_type)

    # base = build_model(model_name)
    mare_upcycling = build_model(
        model_name,
        dataset_type,
        use_mare_upcycling=True,
        num_experts=4,
        num_generalists=2,
    )
