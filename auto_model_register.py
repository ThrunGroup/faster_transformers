from experiments.models.sparse_mistral.sparsification_sftt import (
    MistralSparseSiluMLP,
    SparseMistralforCausalLM,
    SparseMistralConfig,
)
from experiments.mistral_instruct_tuning import prepare_sparse_model
import os
from transformers import (
    MistralConfig,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    MistralForCausalLM,
)
from utils.constants import MISTRAL_7B

# AutoConfig.register("sparse_mistral", SparseMistralConfig)
# AutoModelForCausalLM.register(SparseMistralConfig, SparseMistralforCausalLM)

# path = "/scr/jay/ckpt/Mistral_Sparse_refined_web_70p"
# model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,)
# config = AutoConfig.from_pretrained(path, trust_remote_code=True,)
# print(config)
# print(model)
# for m in model.model.layers:
#     print(m.mlp.dead_threshold)
#
#

prefix = "/scr/jay/ckpt/Mistral_Sparse_refined_web"


for sparsity in ["50p", "70p", "90p"]:
    path = prefix + f"_{sparsity}"
    if os.path.exists(path):
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        config.use_relu = False
        config.save_pretrained(path)

# for sparsity in ["relu"]:
#     path = prefix + f"_{sparsity}"
#     if os.path.exists(path):
#         config = AutoConfig.from_pretrained(path, trust_remote_code=True)
#         config.use_relu = False
#         config.save_pretrained(path)

# path = "/scr/jay/ckpt/sparse_models/unit_test"
# #
# model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
# config = AutoConfig.from_pretrained(path, trust_remote_code=True)
# # model, tokenizer, config = prepare_sparse_model(False, use_lora=False)
# #
# for m in model.model.layers:
#     m.mlp.dead_threshold = 100
# thresholds = [float(m.mlp.dead_threshold) for m in model.model.layers]
# model.config.thresholds = thresholds
# #
# print(config)
#
# # model.save_pretrained(path)
# # tokenizer.save_pretrained(path)
# # model.config.save_pretrained(path)
#
# a = tokenizer("hello world", return_tensors="pt")
# (model(**a))
