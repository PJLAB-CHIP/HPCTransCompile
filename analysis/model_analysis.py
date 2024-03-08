import transformers
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer
)
import torch

class CONFIG:
    base_model_path = '/code/model/CodeLlama-13b-hf'
    lora_model_path = '/code/LLM4HPCTransCompile/mymy/save_model/CodeLlama-13b-hf-600-v3.0_topi_without_ir_shuffle'

class LayerInfo():
    def __init__(self) -> None:
        self.base_t = None
        self.lora_t = None

    def get_diff(self) -> float:
        assert self.base_t.shape == self.lora_t.shape
        diff_t = torch.abs(self.base_t-self.lora_t)
        self.diff_t = diff_t
        self.diff = torch.sum(self.diff_t)
        return self.diff


if __name__ == '__main__':
    base_model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=CONFIG.base_model_path
    )
    lora_model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=CONFIG.lora_model_path
    )

    param_dict = {}

    for name,param in base_model.named_parameters():
        if name not in param_dict:
            param_dict[name] = LayerInfo()
            param_dict[name].base_t = param
        else:
            raise ValueError(f'Already have param {name}')
    
    for name,param in lora_model.named_parameters():
        if name in param_dict:
            param_dict[name].lora_t = param
        else:
            raise ValueError(f'Base model and lora model failed to match on {name}')
    
    for key in param_dict:
        param_dict[key].get_diff()
        print(f'{key}:{param_dict[key].get_diff()}')
    



    # print('base_model:', base_model)
    # print('lora_model:', lora_model)

