from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)

from utils.constants import MISTRAL_7B

pretrained_model_name = "Mistral-7B-v0.1_cola_loading_test"
model_type = MISTRAL_7B

config = AutoConfig.from_pretrained(model_type)

model = AutoModelForSequenceClassification.from_pretrained(
    "thrunlab/" + pretrained_model_name, config=config
)

for name, param in model.named_parameters():
    print(name, param.requires_grad)
