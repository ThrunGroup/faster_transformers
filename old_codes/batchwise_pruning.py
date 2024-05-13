import os

import torch
import lm_eval
import json
import datetime
from accelerate import Accelerator
import transformers

from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, LlamaForCausalLM, MistralForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from experiments.models.cats.cats import Cats, inject_cats, save_to_json, load_data

import torch.nn as nn

# class CatsMistral(MistralForCausalLM):
#     def __init__(self, config):


def cats_training(
    model_type="mistralai/Mistral-7B-v0.1",
    batch_size=1,
    use_batchwise_pruning=True,
    use_datawise_pruning=False,
):
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_type,
    #     torch_dtype=torch.float16,
    #     attn_implementation="flash_attention_2",
    # )  # "meta-llama/Meta-Llama-3-8B")
    sparsity = 0.5  # Hard-coded
    lm_obj = HFLM(
        model_type,
        batch_size=batch_size,
        is_inject_cats=use_batchwise_pruning,
        sparsity=sparsity,
        # attn_implementation="flash_attention_2",
    )
    # model_args = f"max_batch_size={batch_size},is_inject_cats={use_batchwise_pruning},sparsity={sparsity},attn_implementation=flash_attention_2"
    if use_batchwise_pruning:
        inject_cats(lm_obj.model)
    datasets = [
        "openbookqa",
        "arc_easy",
        "winogrande",
        "hellaswag",
        "arc_challenge",
        "piqa",
        "boolq",
        "sciq",
    ]
    results = lm_eval.simple_evaluate(  # call simple_evaluate
        model=lm_obj,
        tasks=datasets,
        num_fewshot=0,
        max_batch_size=batch_size,
        batch_size=batch_size,
        log_samples=False,
        # limit=0.05,
    )

    def log():
        # Get the current date
        current_date = datetime.date.today()

        # File path for the JSON data
        file_path = f"inference_results/zero_shots_results.json"

        # Load existing data
        data = load_data(file_path)

        store_key = (
            f"{model_type}_{current_date}_bsp_{use_batchwise_pruning}_bs{batch_size}"
        )
        data[store_key] = results["results"]
        save_to_json(data, file_path)

    lm_obj.accelerator: Accelerator
    if torch.cuda.device_count() > 1:
        if lm_obj.rank == 0 and (results != None):
            lm_obj.accelerator.print(results["results"])
            model = lm_obj.model
            for name, module in model.named_modules():
                if isinstance(module, Cats):
                    print(
                        f"Module {name} Sparsity: {module.num_zeros/module.num_params}"
                    )
            log()
    else:
        print(results["results"])
        model = lm_obj.model
        for name, module in model.named_modules():
            if isinstance(module, Cats):
                print(f"Module {name} Sparsity: {module.num_zeros/module.num_params}")
        log()


if __name__ == "__main__":
    for model_type in [
        # "facebook/opt-350m",
        "mistralai/Mistral-7B-v0.1",
        # "astronomer/Llama-3-8B-Special-Tokens-Adjusted",
        # "meta-llama/Meta-Llama-3-8B",
    ]:  # "mistralai/Mistral-7B-v0.1",
        for use_bsp in [True]:
            cats_training(
                model_type=model_type, use_batchwise_pruning=use_bsp, batch_size=1
            )
        #     cats_training(
        #         model_type=model_type, use_batchwise_pruning=use_bsp, batch_size=1
        #     )
        # cats_training(model_type=model_type, use_batchwise_pruning=False, batch_size=32)
