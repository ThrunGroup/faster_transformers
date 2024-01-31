# 1. Replace the MistralMLP with mistral_direct_mlp_triton
# 2. Load the per-layer thresholds from a json file
# 3. Pass the thresholds to the MLP layers
# 4. Construct a chat-box use-case with various generation length and context length
# 5. Test the token generation speed

import torch

import json
import argparse
from kernels import mistral_mlp_sparse_direct_flag
from transformers.models.mistral.modeling_mistral import (
    MistralMLP,
)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MistralForCausalLM,
    MistralConfig,
)

class PrunedMistralMLP(MistralMLP):
    def __init__(
            self,
            config,
    ):
        super().__init__(config)
        self.fc2_weight_t = self.register_buffer("fc2_weigth_t", None)
    
    def forward(self, x):
        if (x.shape[1] != 1): # batch size > 1
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        x = x.squeeze(0)
        x_1 = self.act_fn(self.gate_proj(x))
        if self.fc2_weight_t is None:
            self.fc2_weight_t = self.down_proj.weight.t().contiguous()
        x_2 = mistral_mlp_sparse_direct_flag(
            x, # 0 is the beam dimension
            x_1,
            self.up_proj.weight,
            self.fc2_weight_t,
            torch.abs(x_1) > self.threshold,
        )
        return x_2.unsqueeze(0)

def transpose_weights(model):
    for layer in model.model.layers:
        assert isinstance(layer.mlp, PrunedMistralMLP)
        layer.mlp.fc2_weight_t = layer.mlp.down_proj.weight.t().contiguous()

def load_thresholds(model, path):
    with open(path, "r") as f:
        thresholds = json.load(f)
    for i, layer in enumerate(model.model.layers):
        assert isinstance(layer.mlp, PrunedMistralMLP)
        layer.mlp.threshold = thresholds[str(i)]

def replace_mlp(model, config):
    for layer in model.model.layers:
        layer.mlp = PrunedMistralMLP(config).to(torch.float16).to("cuda").eval()

# https://github.com/huggingface/transformers/issues/22222
def generate_sentence(tokenizer, model, prompt: str, prompt_tokens: int = 64, max_tokens: int = 128, num_return_sequences: int = 3, num_beams: int = 4):
    # generate
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").input_ids[:, :prompt_tokens].to(model.device)
    max_new_tokens = max_tokens - inputs.shape[1] - 1
    print("Input length: {}".format(inputs.shape[1]))
    print("Max new tokens: {}".format(max_new_tokens))
    assert max_new_tokens > 0, "max_tokens should be larger than the length of the prompt"
    print(inputs.shape)
    output_ids = model.generate(inputs, do_sample=True, max_new_tokens=max_new_tokens, pad_token_id=2, num_return_sequences=num_return_sequences, temperature=0.7, num_beams=num_beams)
    print(output_ids.shape)
    input_sentence = tokenizer.batch_decode(inputs, skip_special_tokens=True)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return input_sentence, outputs, output_ids.shape[-1]

def benchmark_generation(context_length, generation_length, prompt_path="text.txt", is_debugging=False, use_sparse_model=False, ths_path="thresholds.json"):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if is_debugging:
        config = MistralConfig(
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            # num_attention_heads=4,
        )
        model = MistralForCausalLM(config=config).to(torch.float16).to("cuda").eval()
    else:
        config = MistralConfig()
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
        ).to(torch.float16).to("cuda").eval()

    if use_sparse_model:
        replace_mlp(model, config)
        load_thresholds(model, ths_path)
        transpose_weights(model)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    f = open(prompt_path, "r", encoding='utf-8')
    full_string = f.read()
    f.close()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    input_sentence, output, outlen = generate_sentence(tokenizer, model, full_string, prompt_tokens=context_length, max_tokens=generation_length, num_return_sequences=1, num_beams=1)
    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(input_sentence)
    print("=====================================")
    for i, sentence in enumerate(output):
        print(i)
        print("=====================================")
        print(sentence)
    print("=====================================")
    print("Use sparse? {}; Context Length: {}; Generation Length: {}. Elapsed time: {} ms. Throughput: {:.2f} tokens/s".format(use_sparse_model, context_length, outlen - context_length, elapsed_time_ms, outlen / (elapsed_time_ms / 1000.0)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sparse Mistral inference benchmark")
    parser.add_argument("--context_length", type=int, default=64)
    parser.add_argument("--generation_length", type=int, default=128)
    parser.add_argument("--prompt_path", type=str, default="text.txt")
    parser.add_argument("--is_debugging", action="store_true")
    parser.add_argument("--use_sparse_model", action="store_true")
    parser.add_argument("--ths_path", type=str, default="thresholds.json")
    args = parser.parse_args()
    # benchmark_generation(64, 128, "text.txt", False, True, "thresholds.json")
    benchmark_generation(args.context_length, args.generation_length, args.prompt_path, args.is_debugging, args.use_sparse_model, args.ths_path)