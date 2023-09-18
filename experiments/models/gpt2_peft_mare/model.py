from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
from transformers.pytorch_utils import Conv1D
from transformers.activations import ACT2FN

from experiments.models.MARELoRA import Linear
from experiments.models.router import Router


class MAREMLP(GPT2MLP):
    def __init__(
        self, intermediate_size, config, num_all_rounders, num_experts
    ):
        super().__init__(intermediate_size, config)
        embed_dim = config.hidden_size
        self.c_fc = Linear(
            embed_dim,
            intermediate_size,
            r=4,
            num_experts=num_experts,
            num_all_rounders=num_all_rounders,
        )
        self.act = ACT2FN[config.activation_function]
        self.c_proj = Linear(
            intermediate_size,
            embed_dim,
            r=4,
            num_experts=num_experts,
            num_all_rounders=num_all_rounders,
        )
        self.dropout = nn.Dropout(config.resid_pdrop)
