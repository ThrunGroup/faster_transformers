from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5DenseActDense
import torch
import torch.nn as nn



class MLPSparsityCheck(T5DenseActDense):
    def __init__(self, config: T5Config, block_idx: int):
        super().__init__(config)
        self.config = config
        self.num_zeros = 0
        self.num_params = 0
        self.block_idx = block_idx
        self.counts = 0
        self.counts_per_idx = torch.zeros(config.d_ff)


    def forward(self, hidden_states):
        self.counts += 1
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        self.num_zeros += int(torch.sum(hidden_states==0))
        self.num_params += int(torch.numel(hidden_states))
        self.counts_per_idx = self.counts_per_idx.to(hidden_states.device)
        self.counts_per_idx += torch.sum(hidden_states!=0, dim=(0, 1))
        if self.counts % 100 == 0:
            print(f"block idx: {self.block_idx} / sparsity:{self.num_zeros/self.num_params}")
            a = self.counts_per_idx.clone().detach().sort()
            print(self.counts_per_idx.shape)
            print(f"block idx: {self.block_idx}/ sorted counts: {a[:10], a[-10:]}")
            # print(f"block idx: {self.block_idx} / sparsity:{self.num_negatives / self.num_params}")
            # print(f"block idx: {self.block_idx} / sparsity:{self.num_positives / self.num_params}")
        hidden_states = self.dropout(hidden_states)
        if (
                isinstance(self.wo.weight, torch.Tensor)
                and hidden_states.dtype != self.wo.weight.dtype
                and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states

def apply_sparsity_check_MLP(
        model,
        config,
):
    for is_decoder, network in enumerate(
        [model.transformer.encoder, model.transformer.decoder]
    ):
        for block_idx, block in enumerate(network.block):
            if is_decoder:
                mlp_layer = block.layer[2]
            else:
                mlp_layer = block.layer[1]

            new_mlp = MLPSparsityCheck(config, block_idx)
            old_mlp = mlp_layer.DenseReluDense
            new_mlp.wi.weight.data = old_mlp.wi.weight.data
            new_mlp.wo.weight.data = old_mlp.wo.weight.data

            mlp_layer.DenseReluDense = new_mlp