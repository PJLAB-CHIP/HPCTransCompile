# -*- coding: utf-8 -*-
"""
LLM 微调结果推理
"""
import os, tqdm
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel
import yaml
# import wandb
import argparse
import codebleu
import json
import re


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def cal_codebleu(prediction,reference):
    result = codebleu.calc_codebleu([reference],[prediction],lang='c',weights=(0.25,0.25,0.25,0.25),tokenizer=None)
    print(result)
    return result

def extract_segment(file_content, remove):
    # Match the 'void' function and capture until the closing '}' of the function
    inst_pattern = re.compile(r'(<s>\[INST\].*?\[/INST\])', re.DOTALL)
    inst_matches = inst_pattern.findall(file_content)
    if len(inst_matches) > 1:
        file_content = file_content.split(inst_matches[1], 1)[1]
    pattern = re.compile(r'(void\s+\w+\s*\([^)]*\)\s*\{)', re.DOTALL)
    match = pattern.search(file_content)
    if match:
        start = match.start()
        brace_count = 0
        for i in range(start, len(file_content)):
            if file_content[i] == '{':
                brace_count += 1
            elif file_content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return file_content[start:i+1]
    return file_content.lstrip(remove)

parser = argparse.ArgumentParser()
parser.add_argument('--project_name', default='HPCEval finetune', type=str, help='name')
parser.add_argument('--config', type=str, default='ft_codellama.yaml', help='Path to the config file')
args = parser.parse_args()

config_dir = './exp/'
with open(config_dir + args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = dict2namespace(config)

# 量化配置
bnb_config = BitsAndBytesConfig(
                load_in_4bit = config.bnb.q4bit,
                bnb_4bit_quant_type = config.bnb.quant_type,
                bnb_4bit_use_double_quant = config.bnb.double_quant,
                bnb_4bit_compute_dtype = torch.bfloat16
                )
result_dir = os.path.join(config.model.result_dir, config.model.model_name) 
os.makedirs(result_dir, exist_ok=True)

################################
###  Finetune resluts test  ####
################################
def load_eval(input_file_1, input_file_2, input_file_3=None):
    test_data = []
    with open(input_file_1, 'r') as f:
        data_1 = json.load(f)
    with open(input_file_2, 'r') as f:
        data_2 = json.load(f)
    data = data_1 + data_2 
    if input_file_3 is not None:
        with open(input_file_3, 'r') as f:
            data_3 = json.load(f)
        data = data + data_3
    # print("singl op length:",len(data_1))
    # print("topology op length:",len(data_2))

    # print("benchmark data length:",len(data))

    sorted_data = sorted(data,key=lambda x: len(x['c_code']))
    for i, _data in enumerate(sorted_data):
        print(i,len(_data['c_code']))
    for item in data:
        prompt = f"<s>[INST]{item['cuda_code']}\nTranslate the above CUDA to C, the input tensor shape is {item['input_shape']}. Print only the C code function implementation, end with comment '|End-of-Code'.\n[/INST]"
        # prompt = f"{item['cuda_code']}\nTranslate the above CUDA to C, the input tensor shape is {item['input_shape']}. Print only the C code function implementation, end with comment '|End-of-Code'."
        
        target = item["c_code"]
        name = item["op_name"]
        args = item["op_args"]
        shape = item["input_shape"]
        test_data.append({"instruction": prompt, "gt": target, "op_name":name, "op_name":name, "op_args":args, "input_shape":shape})
    return test_data

def infer_merge_llama_lora():
    """
    合并原始模型和LoRA后进行推理
    """
    test_dir_single = config.data.test_dir_single
    test_dir_topology = config.data.test_dir_topology
    test_dir_model = config.data.test_dir_model

    # test_dataset = load_eval(test_dir_single, test_dir_topology, test_dir_model)
    test_dataset = load_eval(test_dir_single, test_dir_topology)

    # 基础模型路径
    model_path = os.path.join(config.model.model_hub_dir, config.model.model_name) 
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path if config.model.cache_dir is False else "codellama/CodeLlama-13b-hf",
        # model_path ,
        cache_dir = config.model.cache_dir if config.model.cache_dir is not None else None,
        # quantization_config = bnb_config if config.model.use_quant else None,
        device_map = config.model.device_map,
        trust_remote_code = True
    )

    # 微调lora模型路径
    if config.inference.lora_finetune:
        if config.inference.checkpoint:
            lora_model = os.path.join(os.path.join(config.training.output_dir, config.model.model_name), "checkpoint-1200")
        else:
            lora_model = os.path.join(config.model.lora_save_model, config.model.model_name)

        # 加载 base model & lora
        merge_model = PeftModel.from_pretrained(model, lora_model, device_map="auto")  
        # Iterate over the layers
        dir_name = '/code/LLM4HPCTransCompile/weights/codellama/'
        os.makedirs(dir_name, exist_ok=True)
        for w_name, layer in merge_model.named_parameters():
            # 保存数组到 .npy 文件
            w_name = w_name.replace('base_model.model.model.', 'model.')
            w_name = w_name.replace('base_model.model.', '')
            w_name = w_name.replace('.default.', '.')
            w_name = w_name.replace('.base_layer.', '.')
            print(f'Layer {w_name}, shape {layer.shape}')
            f_name = os.path.join('weights/codellama/', f'{w_name}.npy')
            w_tensor = layer.detach()
            # w_tensor = w_tensor.to(torch.float16)
            np.save(f_name, w_tensor.cpu().numpy())

        # 合并
        model = merge_model.merge_and_unload()

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if config.model.cache_dir is False else "codellama/CodeLlama-13b-hf",
        # model_path,
        cache_dir = config.model.cache_dir if config.model.cache_dir is not None else None, 
        trust_remote_code=True
        )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
 
    avg_codebleu = 0
    for input in tqdm.tqdm(test_dataset,total=len(test_dataset)):
        input_ids = tokenizer.encode(input["instruction"], return_tensors="pt",padding=True).to('cuda')
        # print(input_ids)
        input_len = input_ids.numel()
        inference_max_length = 2*input_len + 1024
        # print("len input:", input_len)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, 
                # max_length=config.inference.max_length,
                max_length=inference_max_length,
                do_sample=False,
                num_return_sequences=1,
                num_beams=1.,
                temperature=1.0,
                top_p=1.0,
            )
            print(outputs)
        generate = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generate)
        # pred = generate.lstrip(input["instruction"])
        pred = extract_segment(generate, input["instruction"])
        print(pred)
        gt = input["gt"]
        score = cal_codebleu(pred, gt)
        avg_codebleu += score['codebleu']

        categories = { 'Elementwise':['erf', 'leaky_relu', 'sqrt', 'asin', 'tanh', 'floor', 'log', 'sin', 'prelu', 'fast_exp', 'log2', 'sign', 'sigmoid', 'isnan', 'relu', 'cos', 'fast_tanh', 'log10', 'round', 'atan', 'negative', 'add', 'tan', 'atanh', 'acos', 'ceil', 'asinh', 'abs', 'exp','rsqrt', 'sinh', 'cosh', 'combination_op', 'fast_erf'],
                    'Reduction':['min', 'global_pool_max', 'global_pool_avg', 'sum', 'max', 'rms_norm', 'log_softmax', 'softmax', 'fast_softmax', 'softmax_common', 'prod', 'pool3d', 'pool1d', 'pool2d', 'adaptive_pool_max', 'adaptive_pool_avg'], 
                    'Layout Transform':['reshape', 'transpose', 'gather_nd', 'scatter_nd', 'reorg','unpack_NCHWc_to_nchw', 'flatten', 'scale_shift_nchw', 'flip', 'depth_to_space', 'batch_to_space_nd', 'strided_slice', 'space_to_depth', 'scale_shift_nchwc', 'mirror_pad', 'dilate'], 
                    'Logic Intensive':['fifo_buffer', 'multi_out_op', 'shape', 'upsampling','resize2d', 'resize3d', 'grid_sample', 'argsort'], 
                    'Compute Intensive':['lrn', 'matmul','conv2d_opt', 'dft', 'group_conv2d_opt', 'batch_matmul_opt'] 
                     }
        
        if input["op_name"] == 'topology_expansion':
            show_dir = os.path.join(result_dir, 'topology_expansion')
        else:
            category_found = False
            for cat, ops in categories.items():
                if input["op_name"] in ops:
                    category = cat
                    show_dir = os.path.join(result_dir, f'single/{category}')
                    category_found = True
                    break
            if not category_found:
                show_dir = os.path.join(result_dir, 'model')

        os.makedirs(os.path.join(show_dir, 'pred'), exist_ok=True)
        os.makedirs(os.path.join(show_dir, 'gt'), exist_ok=True)
        os.makedirs(os.path.join(show_dir, 'full_gen'), exist_ok=True)

        with open(os.path.join(show_dir,'pred',"{}_{}_{}.c".format(input["op_name"], input["op_args"], input["input_shape"])),'w') as file:
            file.write(pred)
        with open(os.path.join(show_dir,'gt',"{}_{}_{}.c".format(input["op_name"], input["op_args"], input["input_shape"])),'w') as file:
            file.write(gt)
        with open(os.path.join(show_dir,'full_gen',"{}_{}_{}.c".format(input["op_name"], input["op_args"], input["input_shape"])),'w') as file:
            file.write(generate)

    with open(os.path.join(result_dir,'codebleu.txt'),'w') as file:
        file.write('average_codebleu:{}\n'.format(avg_codebleu/len(test_dataset)))
 
 
if __name__ == '__main__':
    # sequences_info = load_eval()  # 数据加载
    infer_merge_llama_lora()