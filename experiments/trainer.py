import os
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoTokenizer,
)
from torch.nn import functional as F
import wandb
import copy

import gc
import torch

from data.get_dataset import get_dataset
from models.build_model import build_model
from experiments.models.t5_sparity_check.t5_sparsity_check_original import (
    MLPRouter,
)
from utils.constants import (
    # Models
    GPT2,
    GPT2_XL,
    T5,
    T5_3B,
    # Datasets
    # SST2,
    BILLSUM,
    SQUAD,
    QNLI,
    SST2,
    COLA,
    WIC,
    MULTIRC,
    BOOLQ,
)


class ReLUTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(MARETrainer, self).__init__(*args, **kwargs)
        self.steps = 0
        self.relu_loss = torch.nn.MSELoss()

    def compute_penalty(self, model):
        loss = None
        self.steps += 1
        count = 0

        for is_decoder, network in enumerate(
            [
                model.transformer.encoder,
                model.transformer.decoder,
            ]
        ):
            for block_idx, block in enumerate(network.block):
                if is_decoder:
                    mlp_layer = block.layer[2].DenseReluDense
                    # continue
                else:
                    # if block_idx < 6:
                    #     continue
                    mlp_layer = block.layer[1].DenseReluDense

                if mlp_layer.__class__.__name__ != "MLPSparsityCheck":
                    continue

                count += 1

                if (
                    mlp_layer.routing_probs is None
                    or mlp_layer.activation is None
                ):
                    continue

                layer_loss = self.router_loss(
                    mlp_layer.routing_probs,
                    mlp_layer.activation,
                )

                if loss is None:
                    loss = layer_loss
                else:
                    loss += layer_loss

        if loss is not None:
            loss = loss / count

        if self.steps % 50 == 0:
            print(loss)

        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")[:, 0]
        labels = labels.to(logits.dtype)

        # # get penalty term
        # lambda_coeff = 1
        # regularizer = lambda_coeff * self.compute_penalty(model)
        # loss = F.binary_cross_entropy(logits, labels)
        alpha = 1.0
        # penalty = self.compute_penalty(model)
        penalty = None
        loss = F.binary_cross_entropy(logits, labels)

        if penalty:
            loss = loss + alpha * penalty

        return (loss, outputs) if return_outputs else loss


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)

        logits = outputs.get("logits")[:, 0]
        labels = labels.to(logits.dtype)

        # Check if there are any NaN values in logits
        if torch.isnan(logits).any():
            # If there are NaN values, return a zero loss for this batch
            # This effectively "skips" optimizing this batch.
            # The zero gradients won't affect the optimizer update.
            print("NAN")
            loss = torch.tensor(0.0).to(logits.device).requires_grad_()
        else:
            loss = F.binary_cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


class MARETrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(MARETrainer, self).__init__(*args, **kwargs)
        self.steps = 0
        self.router_loss = torch.nn.MSELoss()

    def compute_penalty(self, model):
        loss = None
        self.steps += 1
        count = 0

        for is_decoder, network in enumerate(
            [
                model.transformer.encoder,
                model.transformer.decoder,
            ]
        ):
            for block_idx, block in enumerate(network.block):
                if is_decoder:
                    mlp_layer = block.layer[2].DenseReluDense
                    # continue
                else:
                    # if block_idx < 6:
                    #     continue
                    mlp_layer = block.layer[1].DenseReluDense

                if mlp_layer.__class__.__name__ != "MLPSparsityCheck":
                    continue

                count += 1

                if (
                    mlp_layer.routing_probs is None
                    or mlp_layer.activation is None
                ):
                    continue

                layer_loss = self.router_loss(
                    mlp_layer.routing_probs,
                    mlp_layer.activation,
                )

                if loss is None:
                    loss = layer_loss
                else:
                    loss += layer_loss

        if loss is not None:
            loss = loss / count

        if self.steps % 50 == 0:
            print(loss)

        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")[:, 0]
        labels = labels.to(logits.dtype)

        # # get penalty term
        # lambda_coeff = 1
        # regularizer = lambda_coeff * self.compute_penalty(model)
        # loss = F.binary_cross_entropy(logits, labels)
        alpha = 1.0
        # penalty = self.compute_penalty(model)
        penalty = None
        loss = F.binary_cross_entropy(logits, labels)

        if penalty:
            loss = loss + alpha * penalty

        return (loss, outputs) if return_outputs else loss
