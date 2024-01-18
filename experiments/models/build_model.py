from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)
import torch
from torch import nn
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

from experiments.models.sparse_mistral.relufication import (
    apply_relufication,
    apply_mistral_mlp_stats,
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
)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def set_up_model_for_lora(model):
    for name, param in model.named_parameters():
        if "lora" in name or "score" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()
    return model


def get_lora_model(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "v_proj", "gate_proj"],
    )

    model = set_up_model_for_lora(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def build_model(
    model_type: str,
    use_mare_mlp: bool = False,
    capacity_factor: int = 0,
    num_generalists: int = 0,
    num_experts: int = 0,
    expert_size: int = 0,
    pretrained_model_name: str = None,
    use_relu: bool = False,
):
    print("Loading the model...")
    config = AutoConfig.from_pretrained(model_type)

    if pretrained_model_name:
        print("Loading from thrunlab/" + pretrained_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            "thrunlab/" + pretrained_model_name, config=config
        )
        model = set_up_model_for_lora(model)

        apply_mistral_mlp_stats(
            model, config, max_dead_threshold=0.2
        )  # each MLP layer can collect stats
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_type, config=config
        )
        # print(model)
        apply_mistral_mlp_stats(
            model, config, max_dead_threshold=0.2
        )  # each MLP layer can collect stats

        print("Applying LoRA...")
        model = get_lora_model(model)

    if use_relu:
        apply_relufication(model)

    if use_mare_mlp:
        apply_sparsity_check_MLP(
            model,
            config,
            num_generalists=num_generalists,
            num_experts=num_experts,
            expert_size=expert_size,
        )

    return model


if __name__ == "__main__":
    model_name = T5
    model_name = "lukeleeai/t5-base_baseline"
    dataset_type = BILLSUM

    mare_upcycling = build_model(
        model_name,
        dataset_type,
        use_mare_upcycling=True,
        num_experts=8,
        num_generalists=1,
        capacity_factor=2,
    )

    print(mare_upcycling)
