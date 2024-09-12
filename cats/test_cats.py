import unittest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tempfile import TemporaryDirectory
import os

# Assuming the CatsModel and related classes are in a file named cats_model.py
from cats.cats_model import CatsModelForCausalLM, CatsConfig, get_cats_model


class TestCatsModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "mistralai/Mistral-7B-v0.1"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.base_model = AutoModelForCausalLM.from_pretrained(cls.model_name)

    def test_create_cats_model(self):
        cats_model = get_cats_model(
            self.base_model,
            target_modules=["act_fn"],
            target_sparsity=0.5,
            kernel_inject_targets={"mlp": 2},
            is_inject_kernel=False,
            is_share_params=True,
        )
        self.assertIsInstance(cats_model, CatsModelForCausalLM)

    def test_save_and_load_cats_model(self):
        cats_model = get_cats_model(self.base_model, is_share_params=False)

        with TemporaryDirectory() as temp_dir:
            # Save the model
            save_path = os.path.join(temp_dir, "cats_model")
            cats_model.save_pretrained(save_path)

            # Load the model
            loaded_model = CatsModelForCausalLM.from_pretrained(save_path)

            # Check if the loaded model has the same configuration
            # self.assertEqual(cats_model.config.to_dict(), loaded_model.config.to_dict())
            print(cats_model.config, loaded_model.config)

            # Check if the model can process input
            input_ids = self.tokenizer.encode("Hello, world!", return_tensors="pt")
            with torch.no_grad():
                original_output = cats_model(input_ids)
                loaded_output = loaded_model(input_ids)

            # Check if the outputs are the same
            torch.testing.assert_close(original_output.logits, loaded_output.logits)

    def test_inference(self):
        cats_model = get_cats_model(self.base_model, is_share_params=False)
        input_text = "Hello, world!"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        with torch.no_grad():
            output = cats_model.generate(input_ids, max_length=50)

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        self.assertIsInstance(generated_text, str)
        self.assertGreater(len(generated_text), len(input_text))


if __name__ == "__main__":
    unittest.main()
