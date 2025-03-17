# Patching the Gradient Accumulation bug in Huggingface Transformers by monkey patching _inner_training_loop function

from llava.train.llama_grad_accum_monkey_patch import (
    replace_transformers_trainning_loop_with_grad_accum_pacthed_trainning_loop,
)
replace_transformers_trainning_loop_with_grad_accum_pacthed_trainning_loop()

from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation='flash_attention_2')
