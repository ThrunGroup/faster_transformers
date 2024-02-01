import os.path

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
import torch
from torch import nn
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

from experiments.models.sparse_mistral.relufication import (
    apply_mistral_sparse_silu_mlp,
)


from utils.constants import (
    # models
    T5,
    # datasets
    BILLSUM,
)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def set_up_model_for_lora(model):
    """
    Use this function to a model finetuned with LoRA.
    @luke: Maybe there is a cleaner way to do this, but I'm leaving it for now.
    """
    for name, param in model.named_parameters():
        if "lora" in name or "score" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    model.enable_input_require_grads()
    return model


def get_lora_model(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "v_proj", "gate_proj", "down_proj"],
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
    use_sparse_silu: bool = False,
    use_lora: bool = True,
):
    print("Loading the model...")
    config = AutoConfig.from_pretrained(model_type)

    if pretrained_model_name:
        # Load a finetuned model
        print("Loading from thrunlab/" + pretrained_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            "thrunlab/" + pretrained_model_name, config=config
        )
        model = set_up_model_for_lora(model)

        # As there is a bug with loading a score head from huggingface repository,
        # load it from a local file if it exists
        score_path = f"scores/{pretrained_model_name}.pt"
        if os.path.exists(score_path):
            score_dict = torch.load(f"scores/{pretrained_model_name}.pt")
            model.score.load_state_dict(score_dict)
    else:
        # Load a pretrained model not finetuned on a specific downstream task
        # model = AutoModelForSequenceClassification.from_pretrained(
        #     model_type, config=config
        # )
        model = AutoModelForCausalLM.from_pretrained(model_type, config=config)

        # You can optionally uncomment this block
        # to load a finetuned score head for faster finetuning
        # pretrained_model_name = "Mistral-7B-v0.1_cola_original2"
        # score_dict = torch.load(f"scores/{pretrained_model_name}.pt")
        # model.model.score.load_state_dict(score_dict)

    # Replace a normal MLP layer with a MistralSparseSiluMLP layer.
    # By default, it behaves exactly as a normal MLP layer unless
    # you turn on use_sparse_silu in `create_models.py`
    apply_mistral_sparse_silu_mlp(model, config)

    if use_lora:
        print("Applying LoRA...")
        model = get_lora_model(model)
    # print(model)
    #
    # if not isinstance(model.model.score, nn.Linear):
    #     print("Changing model score")
    #     model.model.score = model.model.score.original_module
    #     print("New score head: ")
    #     print(model.model.score)

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
