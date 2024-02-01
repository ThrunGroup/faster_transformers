# 1. Replace the MistralMLP with mistral_direct_mlp_triton
# 2. Load the per-layer thresholds from a json file
# 3. Pass the thresholds to the MLP layers
# 4. Construct a chat-box use-case with various generation length and context length
# 5. Test the token generation speed

import torch
import torch.nn as nn
from typing import Optional, Tuple

from transformers.activations import ACT2FN

import os
import json
import argparse
from kernels import mistral_mlp_sparse_direct_flag, gather_gemv2_flag, gather_transposed_gemv_flag
from transformers.models.mistral.modeling_mistral import (
    MistralMLP,
    MistralDecoderLayer,
    MistralConfig,
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
            return super().forward(x)# self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
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

class PredictorMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.sp_rank = config.sp_rank
        self.pred_down = nn.Linear(self.hidden_size, self.sp_rank, bias=False)
        self.pred_up = nn.Linear(self.sp_rank, self.intermediate_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]
    
    def forward(self, x):
        return  torch.abs(self.act_fn(self.pred_up(self.pred_down(x)))) > self.threshold

class FlagMistralMLP(MistralMLP):
    def __init__(self, config):
        super().__init__(config)
        self.fc2_weight_t = self.register_buffer("fc2_weigth_t", None)
    
    def forward(self, x: torch.Tensor, flags: torch.Tensor):
        if (x.shape[1] != 1):
            return super().forward(x)
        x = x.squeeze(0)
        if self.fc2_weight_t is None:
            self.fc2_weight_t = self.down_proj.weight.t().contiguous()
        x = gather_gemv2_flag(x, self.gate_proj.weight, self.up_proj.weight, flags, self.config.hidden_act)
        return gather_transposed_gemv_flag(x, self.fc2_weight_t, flags).unsqueeze(0)

class AsyncSparseMistralDecoderLayer(MistralDecoderLayer):
    def __init__(self, config: MistralConfig, layer_id: int):
        super().__init__(config, layer_id)
        self.pred_mlp = PredictorMLP(config)
        self.mlp = FlagMistralMLP(config)
        self.pred_stream = torch.cuda.Stream(device="cuda", priority=0)
        self.event_mlp_sp = torch.cuda.Event(enable_timing=False, blocking=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        curr_stream = torch.cuda.current_stream()

        residual = hidden_states

        with torch.cuda.stream(self.pred_stream):
            flags = self.pred_mlp(residual)
            self.pred_stream.record_event(self.event_mlp_sp)

        flags = self.pred_mlp(residual)

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        curr_stream.wait_event(self.event_mlp_sp)
        hidden_states = self.mlp(hidden_states, flags)
            
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class SparseMistralDecoderLayer(MistralDecoderLayer):
    def __init__(self, config: MistralConfig, layer_id: int):
        super().__init__(config, layer_id)
        self.pred_mlp = PredictorMLP(config)
        self.mlp = FlagMistralMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        flags = self.pred_mlp(residual)

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states, flags)
            
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def store_fake_svd_weights(model, config, path):
    svd_weights = {}
    torch.manual_seed(0)
    for i in range(len(model.model.layers)):
        svd_weights[f"layer{i}.pred_mlp.w1"] = torch.randn(config.sp_rank, config.hidden_size).to(torch.float16)
        svd_weights[f"layer{i}.pred_mlp.w2"] = torch.randn(config.intermediate_size, config.sp_rank).to(torch.float16)
        svd_weights[f"layer{i}.pred_mlp.b2"] = torch.randn(config.intermediate_size).to(torch.float16)
    torch.save(svd_weights, path)

def load_svd_weights(model, path):
    svd_weights = torch.load(path)
    for i, layer in enumerate(model.model.layers):
        layer.pred_mlp.pred_down.weight = torch.nn.Parameter(svd_weights[f"layer{i}.pred_mlp.w1"].to(torch.float16).to("cuda"))
        layer.pred_mlp.pred_up.weight = torch.nn.Parameter(svd_weights[f"layer{i}.pred_mlp.w2"].to(torch.float16).to("cuda"))
        layer.pred_mlp.pred_up.bias = torch.nn.Parameter(svd_weights[f"layer{i}.pred_mlp.b2"].to(torch.float16).to("cuda"))

def transpose_weights(model):
    for layer in model.model.layers:
        assert isinstance(layer.mlp, PrunedMistralMLP) or isinstance(layer.mlp, FlagMistralMLP)
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

def load_threshold_pred(model, ths_path):
    with open(ths_path, "r") as f:
        thresholds = json.load(f)
    for i, layer in enumerate(model.model.layers):
        assert isinstance(layer.pred_mlp, PredictorMLP)
        layer.pred_mlp.threshold = thresholds[str(i)]

def replace_decoder_layer(model, config):
    new_layers = []
    for layer_id in range(config.num_hidden_layers):
        new_layers.append(SparseMistralDecoderLayer(config=config, layer_id=layer_id))
    model.model.layers = nn.ModuleList(new_layers) 

def replace_decoder_layer_async(model, config):
    new_layers = []
    for layer_id in range(config.num_hidden_layers):
        new_layers.append(AsyncSparseMistralDecoderLayer(config=config, layer_id=layer_id))
    model.model.layers = nn.ModuleList(new_layers)   

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

def benchmark_generation(context_length, generation_length, prompt_path="text.txt", is_debugging=False, sparse_mode=0, ths_path="thresholds.json", **kwargs):
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

    if sparse_mode == 0:
        replace_mlp(model, config)
        load_thresholds(model, ths_path)
        transpose_weights(model)
    elif sparse_mode == 1:
        svd_path = kwargs.pop("svd_path", "svd_weights.pth")
        config.sp_rank = kwargs.pop("sp_rank", 128)
        if not os.path.exists(svd_path):
            store_fake_svd_weights(model, config, svd_path)
        replace_decoder_layer(model, config)
        load_threshold_pred(model, ths_path)
        load_svd_weights(model, svd_path)
        model = model.to(torch.float16).to("cuda").eval()
        transpose_weights(model)
    elif sparse_mode == 2:
        svd_path = kwargs.pop("svd_path", "svd_weights.pth")
        config.sp_rank = kwargs.pop("sp_rank", 128)
        if not os.path.exists(svd_path):
            store_fake_svd_weights(model, config, svd_path)
        replace_decoder_layer_async(model, config)
        load_threshold_pred(model, ths_path)
        load_svd_weights(model, svd_path)
        model = model.to(torch.float16).to("cuda").eval()  
        
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
    print("Sparse mode: {}; Context Length: {}; Generation Length: {}. Elapsed time: {} ms. Throughput: {:.2f} tokens/s".format(sparse_mode, context_length, outlen - context_length, elapsed_time_ms, outlen / (elapsed_time_ms / 1000.0)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sparse Mistral inference benchmark")
    parser.add_argument("--context_length", type=int, default=64)
    parser.add_argument("--generation_length", type=int, default=1089)
    parser.add_argument("--prompt_path", type=str, default="text.txt")
    parser.add_argument("--is_debugging", action="store_true")
    parser.add_argument("--sparse_mode", type=int, default=-1)
    parser.add_argument("--ths_path", type=str, default="thresholds.json")
    parser.add_argument("--sp_rank", type=int, default=128)
    parser.add_argument("--svd_path", type=str, default="svd_weights.pth")
    args = parser.parse_args()
    # benchmark_generation(64, 128, "text.txt", False, True, "thresholds.json")
    kwargs = {"sp_rank": args.sp_rank, "svd_path": args.svd_path}
    benchmark_generation(args.context_length, args.generation_length, args.prompt_path, args.is_debugging, args.sparse_mode, args.ths_path, kwargs=kwargs)