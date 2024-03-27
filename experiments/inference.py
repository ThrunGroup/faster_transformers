from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    LlamaForCausalLM,
    LlamaConfig,
)
from torch.utils.data import DataLoader
from utils.constants import LLAMA, REFINED_WEB
from experiments.models.sparse_silu.ugly_utils import *
from experiments.data.get_dataset import get_dataset


def prepare_sparse_model():
    base_model_name = "/matx/u/lukeai/ckpt/sparse_llama_7b_refined_web_50p_no_adapter"
    # base_model_name = "meta-llama/Llama-2-7b"
    BaseConfig = LlamaConfig
    BaseCausalLM = LlamaForCausalLM
    SparseConfig = SparseLlamaConfig
    SparseCausalLM = SparseLlamaForCausalLM
    SparseConfig.register_for_auto_class()
    SparseCausalLM.register_for_auto_class("AutoModelForCausalLM")
    config = BaseConfig.from_pretrained(base_model_name)
    config = get_sparse_config(config)
    config.use_cache = True
    model = SparseCausalLM.from_pretrained(
        base_model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.config_class = SparseConfig
    apply_sparse_silu_mlp(model, model.config, use_sparse_regularization=False)
    enable_sparse_silu(model)
    model.config.use_sparse_predictor = False
    model.config.use_relu = False
    model.config.use_sparse_model = True
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def sparse_inference():
    act_hist_path = "/matx/u/zgh23/act_hist/sparse_llama_7b_hf/refined_web.pt"
    device = torch.device("cuda")
    dataset_type = REFINED_WEB
    model_type = LLAMA
    model, tokenizer = prepare_sparse_model()
    activate_stats(model)
    load_act_hist(model, act_hist_path)
    set_sparse_threshold(model, 0.5)
    deactivate_stats(model)
    dataset = get_dataset(dataset_type, tokenizer, model_type, max_seq_length=1000)
    _, _, test_dataset = dataset.get_tokenized_dataset()
    data_collator = dataset.get_data_collator()
    dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator)
    counts = 0
    # Generate tokens in batch
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)  # Move input tensors to the device
            max_lengths = input_ids.size(1) + 500
            print(max_lengths)
            outputs = model.generate(input_ids=input_ids, max_length=max_lengths, pad_token_id=tokenizer.eos_token_id, use_cache=True)
            # Decode generated tokens to texts
            generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            counts += 1
            if counts <= 1:
                for idx in range(len(input_ids)):
                    ids = input_ids[idx]
                    # Decode the input_ids to a string
                    tokens_str = tokenizer.decode(ids, skip_special_tokens=True)
                    print(f"Tokens before generation: \n {tokens_str}")
                    generated_text = generated_texts[idx]
                    print(f"Generated: \n {generated_text}\n")


if __name__ == "__main__":
    sparse_inference()
