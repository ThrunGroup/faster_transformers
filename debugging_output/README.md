---
tags:
- generated_from_trainer
datasets:
- openwebtext
model-index:
- name: pretraining_test
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# pretraining_test

This model is a fine-tuned version of [](https://huggingface.co/) on the openwebtext dataset.
It achieves the following results on the evaluation set:
- Loss: 10.3700

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 8
- eval_batch_size: 16
- seed: 0
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 2
- total_train_batch_size: 32
- total_eval_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- training_steps: 100

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 10.368        | 25.0  | 50   | 10.3705         |
| 10.3672       | 50.0  | 100  | 10.3700         |


### Framework versions

- Transformers 4.35.2
- Pytorch 2.1.1+cu121
- Datasets 2.15.0
- Tokenizers 0.15.0
