from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)
import torch

from utils.constants import MISTRAL_7B

pretrained_model_name = "Mistral-7B-v0.1_cola_loading_test"
model_type = MISTRAL_7B

config = AutoConfig.from_pretrained(model_type)

model = AutoModelForSequenceClassification.from_pretrained(
    "thrunlab/" + pretrained_model_name, config=config
)

for name, param in model.named_parameters():
    if "lora" in name or "score" in name:
        param.requires_grad = True
        print(name)

score_state_dict = torch.load(f"scores/{pretrained_model_name}1.pt")
print("original score: ", model.score.weight)
model.score.load_state_dict(score_state_dict)
print("Loaded score  : ", model.score.weight)
