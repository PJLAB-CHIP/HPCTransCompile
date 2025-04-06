import torch
from os.path import join,exists
import os
from tqdm import tqdm
import argparse
import json
#### Prompt ####
from AI_CUDA_Engineer.prompt_generator import PromptGenerator
#### Utils ####
from utils.api_utils import api_call,local_call
from utils.utils import load_prompt,save_to_file

def parse_args():
    parser = argparse.ArgumentParser(description='Generated target platform code.')
    parser.add_argument('--device',type=str)
    parser.add_argument('--model_name',type=str)
    parser.add_argument('--level',type=str)
    parser.add_argument('--action',type=str)
    parser.add_argument('--range',type=str,default='left')
    parser.add_argument('--use_lora',type=bool,default=False)
    return parser.parse_args()

args = parse_args()

device = args.device if torch.cuda.is_available() else 'cpu'

def no_pass_generate_batch(model_name,prompt_generator:PromptGenerator):
    no_pass_dict = {
        'level1':['54_conv_standard_3D__square_input__square_kernel',
                  '59_conv_standard_3D__asymmetric_input__square_kernel',
                  '61_conv_transposed_3D__square_input__square_kernel',
                  '66_conv_standard_3D__asymmetric_input__asymmetric_kernel',
                  '68_conv_transposed_3D__square_input__asymmetric_kernel',
                  '70_conv_transposed_3D__asymmetric_input__square_kernel',
                  '73_conv_transposed_3D_asymmetric_input_square_kernel__strided_padded__grouped',
                  '77_conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__',
                  '92_cumsum_exclusive'],
        'level2':['79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max',
                  '83_Conv3d_GroupNorm_Min_Clamp_Dropout'],
        'level3':['20_MobileNetV2',
                  '21_EfficientNetMBConv',
                  '26_ShuffleNet',
                  '28_VisionTransformer',
                  '29_SwinMLP',
                  '30_SwinTransformerV2',
                  '31_VisionAttention',
                  '32_ConvolutionalVisionTransformer',
                  '38_LTSMBidirectional',
                  '48_Mamba2ReturnY']
    }
    for operator in tqdm(no_pass_dict['level3']):
        # generate(model_name,prompt_generator,'level3',operator,action='conversion')
        generate(model_name,prompt_generator,'level3',operator,action='translation')

    # for key,val in no_pass_dict.items():
    #     print(f'Level: {key}.')
    #     for operator in tqdm(val):
    #         generate(model_name,prompt_generator,key,operator,action='conversion')
    #         generate(model_name,prompt_generator,key,operator,action='translation')

def generate(model_name,prompt_generator:PromptGenerator,level,operator,action,use_lora):
    code = prompt_generator.load_source_code_single(level,operator,action=action,return_code=True,model_name=model_name)
    base_prompt,system_prompt = prompt_generator.fill_template(code,action=action,operator=operator)
    if model_name == 'deepseek':
        content = api_call(model_name,system_prompt,base_prompt)
    elif model_name in ['QwenCoder_14b','DeepSeekCoder_Lite','LingCoder','OpenCoder']:
        content = local_call(model_name,system_prompt,base_prompt,device,use_lora=use_lora)
    
    if action == 'optimization_c':
        if use_lora:
            save_path = join(prompt_generator.save_path_dict[action],f"{model_name}_optim_lora",level)
        else:
            save_path = join(prompt_generator.save_path_dict[action],f"{model_name}_optim",level)
    else:
        if use_lora:
            save_path = join(prompt_generator.save_path_dict[action],f"{model_name}_lora",level)
        else:
            save_path = join(prompt_generator.save_path_dict[action],model_name,level)
    # 临时设置 #
    save_path = join(prompt_generator.save_path_dict[action],f"{model_name}_simplify",level)      
    ###########  
    os.makedirs(save_path,exist_ok=True)
    save_to_file(content,operator,suffix=prompt_generator.suffix_dict[action],action=action,save_path=save_path)

def generate_batch(model_name,prompt_generator:PromptGenerator,level,action,range,use_lora):
    if range == 'left':
        operators = extract_residual_operator(model_name,level)
    elif range == 'all':
        base_path = '/code/LLM4HPCTransCompile/ParallelBench'
        operators = [os.path.splitext(file)[0] for file in os.listdir(join(base_path,level)) if os.path.isfile(join(base_path,level,file))]
    print('====Operators====')
    print(operators)
    for operator in tqdm(operators):
        generate(model_name,prompt_generator,level,operator,action,use_lora)

def extract_residual_operator(model_name,level):
    all_operators_path = join('/code/KernelBench/KernelBench',level)
    all_operators = [os.path.splitext(file)[0] for file in os.listdir(all_operators_path) if os.path.isfile(join(all_operators_path,file))]
    generated_operators_path = join('/code/LLM4HPCTransCompile/Results',model_name,level)
    generated_operators = [os.path.splitext(file)[0] for file in os.listdir(generated_operators_path) if os.path.isfile(join(generated_operators_path,file))]
    return list(filter(lambda x: x not in generated_operators, all_operators))

def extract_optim_operator(model_name,level):
    """
    提取需要进行优化的算子列表
    """
    with open(join('/code/LLM4HPCTransCompile/Results',args.model_name,f'{args.level[-1]}_eval_results.json'),'r',encoding='utf-8') as file:
        data = json.load(file)


if __name__ == '__main__':
    # QwenCoder_14b DeepSeekCoder_Lite CodeShell_7b CodeLlama_34b
    prompt_generator = PromptGenerator()
    generate_batch(args.model_name,prompt_generator,args.level,action=args.action,range=args.range,use_lora=args.use_lora)
    """EXAMPLE1"""
    # no_pass_generate_batch('deepseek',prompt_generator)
    """EXAMPLE2"""
    # level = 'level1'
    # operator = '3_Batched_matrix_multiplication'
    # generate('QwenCoder_14b',prompt_generator,level,operator,action='optimization_c')