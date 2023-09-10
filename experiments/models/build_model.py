import torch
from transformers import GPT2Config, GPT2ForSequenceClassification, GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType

# from .gpt2_gated import apply_gated_attention
from .gpt2_gated_faster import apply_gated_attention
from utils.constants import GPT2, GPT2_MEDIUM

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def build_model(model_name: str, use_lora: bool, use_gated_attention: bool, freeze_router: bool = False,):
    config = GPT2Config.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)

    if use_gated_attention:
        apply_gated_attention(model, config, freeze_router=freeze_router,)

    if use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False,
            r=32, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)

    model.to(device)
    return model


if __name__ == '__main__':
    model_name = GPT2_MEDIUM
    base = build_model(model_name, use_lora=True, use_gated_attention=False)
    lora_base = build_model(model_name, use_lora=True, use_gated_attention=False)
    gated = build_model(model_name, use_lora=True, use_gated_attention=False)
    lora_gated = build_model(model_name, use_lora=True, use_gated_attention=False)

    for (b, lb, g, lg) in zip(base.transformer.h, lora_base.transformer.h,
                              gated.transformer.h, lora_gated.transformer.h):
        match1 = torch.allclose(b.attn.c_attn.weight, lb.attn.c_attn.weight)
        match2 = torch.allclose(b.attn.c_attn.weight, g.attn.c_attn.weight)
        match3 = torch.allclose(b.attn.c_attn.weight, lg.attn.c_attn.weight)
        match4 = torch.allclose(lb.attn.c_attn.weight, g.attn.c_attn.weight)
        match5 = torch.allclose(lb.attn.c_attn.weight, lg.attn.c_attn.weight)
        match6 = torch.allclose(g.attn.c_attn.weight, lg.attn.c_attn.weight)

        if not (match1 and match2 and match3 and match4 and match5 and match6):
            print(b)