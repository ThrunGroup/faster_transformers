import deepspeed

# Set the path to your checkpoint directory.
ckpt_dir = "/scr/lukeai/ckpt/sparse_llama_7b_refined_web_90p/checkpoint-200"

# Use DeepSpeed to load the checkpoint.
# The 'client_state' will contain the state dict if the checkpoint includes it.
_, client_state = deepspeed.checkpointing.load_checkpoint(ckpt_dir, None)

# Assuming that the state dict is stored within client_state under the 'model' key.
state_dict = client_state["model"]

# Now, you can inspect the state_dict as before.
for key in state_dict.keys():
    print(key)

# If you need to see the size of each parameter:
for key, value in state_dict.items():
    print(f"{key}: {value.size()}")
