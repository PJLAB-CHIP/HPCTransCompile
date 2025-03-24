import torch
from os.path import join,exists
import os
from tqdm import tqdm
#### Prompt ####
from AI_CUDA_Engineer.prompt_generator import PromptGenerator
#### Utils ####
from utils.api_utils import api_call,local_call
from utils.utils import load_prompt,save_to_file

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate(model_name,prompt_generator:PromptGenerator,level,operator,action):
    code = prompt_generator.load_source_code_single(level,operator,action=action,return_code=True)
    base_prompt,system_prompt = prompt_generator.fill_template(code,action=action)
    if model_name == 'deepseek':
        content = api_call(model_name,system_prompt,base_prompt)
    elif model_name == 'QwenCoder_14b':
        content = local_call(model_name,system_prompt,base_prompt,device)
    save_path = join(prompt_generator.save_path_dict[action],level)
    os.makedirs(save_path,exist_ok=True)
    save_to_file(content,operator,suffix=prompt_generator.suffix_dict[action],action=action,save_path=save_path)

def generate_batch(model_name,prompt_generator:PromptGenerator,level,action):
    base_path = '/code/LLM4HPC/benchmark/cuda'
    operators = [os.path.splitext(file)[0] for file in os.listdir(join(base_path,level)) if os.path.isfile(join(base_path,level,file))]
    # TODO: 30/91
    for operator in tqdm(operators):
        generate(model_name,prompt_generator,level,operator,action)

if __name__ == '__main__':
    level = 'level1'
    operator = '3_Batched_matrix_multiplication'
    prompt_generator = PromptGenerator()
    # generate('QwenCoder_14b',prompt_generator,level,operator,action='translation_c')
    generate_batch('QwenCoder_14b',prompt_generator,level,action='translation_c')