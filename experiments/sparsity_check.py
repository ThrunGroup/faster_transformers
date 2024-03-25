from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from experiments.data.get_dataset import get_dataset
from experiments.models.sparse_mistral.sparsification_sftt import (
    SparseMistralforCausalLM,
    activate_stats,
    print_dead_neuron_stats,
    enable_sparse_silu,
    SparseMistralConfig,
    get_sparse_mistral_config,
)
from utils.constants import OPENWEBTEXT, MISTRAL_7B


def sparsity_check(model_dir):
    # AutoConfig.register("sparse_mistral", SparseMistralConfig)
    # AutoModelForCausalLM.register(SparseMistralConfig, SparseMistralforCausalLM)

    # config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    # config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    # config = get_sparse_mistral_config(config)
    # model: SparseMistralforCausalLM = AutoModelForCausalLM.from_pretrained(
    #     model_dir,
    #     trust_remote_code=True,
    # )
    config = SparseMistralConfig.from_pretrained(MISTRAL_7B)
    model = AutoModelForCausalLM.from_pretrained(MISTRAL_7B, trust_remote_code=True)
    sparse_config = get_sparse_mistral_config(config)

    SparseMistralforCausalLM.register_for_auto_class(AutoModelForCausalLM)
    SparseMistralConfig.register_for_auto_class(AutoConfig)
    # sparse_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    print(sparse_config)
    print("HELLO")
    sparse_model = SparseMistralforCausalLM(sparse_config)
    sparse_model.load_state_dict(model.state_dict())
    sparse_model.register_for_auto_class(AutoModelForCausalLM)
    sparse_config.register_for_auto_class(AutoConfig)

    print(sparse_model)

    print(sparse_model.__class__)

    # print(model)
    # print(model.__class__)

    model.save_pretrained("/scr/jay/debugging/hi")
    model = AutoModelForCausalLM.from_pretrained("/scr/jay/debugging/hi", trust_remote_code=True)
    print(model.__class__)

    # print(model.__class__.config_class)
    # model.__class__.config_class = SparseMistralConfig
    # print(model.__class__.config_class)

    # tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # model.eval()
    # print(model)
    # print(model.__class__)
    # # for layer in model.model.layers:
    # #     print(layer.mlp.__class__)
    #
    # dataset = get_dataset(OPENWEBTEXT, tokenizer, "SparseMistral")
    # train_dataset, test_dataset = dataset.get_tokenized_dataset()
    # data_collator = dataset.get_data_collator()
    #
    # trainer_config = TrainingArguments(
    #     output_dir="/scr/jay/trash",
    #     per_device_eval_batch_size=1,
    #     hub_model_id=model_dir,
    #     push_to_hub=True,
    # )
    # trainer = Trainer(
    #     model=model,
    #     data_collator=data_collator,
    #     eval_dataset=test_dataset,
    #     args=trainer_config,
    # )
    # trainer.push_to_hub()
    # enable_sparse_silu(model), activate_stats
    # for i, layer in enumerate(model.model.layers):
    #     layer.mlp.kill_sparse_swish_outputs = True
    #     layer.mlp.is_stats = True
    #     # layer.mlp.activate_stats(is_collect_histogram=False)
    #     print(layer.mlp.dead_percentage)
    #     layer.mlp.a = 0
    #
    # print(trainer.evaluate())
    # for i, layer in enumerate(model.model.layers):
    #     print(layer.mlp.dead_percentage)
    #     print(layer.mlp.a)
    #
    # # print_dead_neuros_stats(model)
    # total_sparsity = 0
    # counts = 0
    # for i, layer in enumerate(model.model.layers):
    #     dead_percentage = layer.mlp.dead_percentage * 100
    #     agg_sparsity = layer.mlp.agg_sparsity * 100
    #     print(f"layer {i} sparsity: {dead_percentage:.3f}%")
    #     print(f"layer {i} agg sparsity: {agg_sparsity:.3f}%")
    #     total_sparsity += dead_percentage
    #     counts += 1
    #
    # print(f"Total sparsity: {total_sparsity/counts: .3f}%")


if __name__ == "__main__":
    # sparsity_check("mistralai/Mistral-7B-v0.1")

    # sparsity_check("/scr/jay/ckpt/2024-02-27/Mistral_Sparse_refined_web_70p")
    # sparsity_check("thrunlab/Mistral_Sparse_refined_web_relu_2024-03-01")
    sparsity_check("/scr/jay/ckpt/2024-03-01/Mistral_Sparse_refined_web_relu")
