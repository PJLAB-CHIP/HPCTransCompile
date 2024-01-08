from ast import Tuple
from re import split
from sympy import sequence
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM
)
from transformers.generation import GenerationConfig
import torch
import os
import json
import tqdm
import argparse
import logging
from utils import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

"""命令行解析参数信息"""
parser = argparse.ArgumentParser()
parser.add_argument('--model_name',type=str,help='foundation model')
parser.add_argument('--train_data_version',type=str)
parser.add_argument('--test_data_version',type=str)
parser.add_argument('--ckpt',type=int)
parser.add_argument('--device',type=int,default=0)
parser.add_argument('--use_lora_model',type=bool,default=False)
parser.add_argument('--bits',type=int,default=8)
args = parser.parse_args()  # 解析命令行参数

print('args:', args)

# MODEL_PATH = "./save_model/CodeLlama-7b-hf-500-v1.2"
# EVAL_DATA_PATH = "../preprocess/data/hpc_test_v1.0.json"
# EVAL_PROMPT = "{}\n\n### Input:\n{}\n\n### Response: "
if args.use_lora_model:
    MODEL_PATH = f"./save_model/{args.model_name}-{args.ckpt}-{args.train_data_version}"
else:
    MODEL_PATH = f"../../model/{args.model_name}"
EVAL_DATA_PATH = f"../preprocess/data/hpc_{args.test_data_version}/hpc_test_{args.test_data_version}.json"
EVAL_PROMPT = "{}\n\n### Input:\n{}\n\n### Response: "

def load_eval():
    """
    加载测试集数据
    """
    with open(EVAL_DATA_PATH, 'r') as file:
        eval_datas = json.load(file)
    # print('len(eval_data):', len(eval_data))
    sequences_info = []
    for index,eval_data in enumerate(eval_datas):
        # print(eval_data)
        sequence_info = {}
        sequence = EVAL_PROMPT.format(eval_data['instruction'],eval_data['input'])
        sequence_info["sequence"] = sequence
        sequence_info["gt"] = eval_data['output']
        sequence_info["op_name"] = eval_data['op_name']+"_{}".format(index)
        sequences_info.append(sequence_info)
    return sequences_info


def show_attention_matrix(attention_tuple: Tuple, input_token_num: int, generate_token_num: int):
    attention_list = []
    for lines in attention_tuple:
        attention_list.append(lines[0][0].cpu().float())
        # print('--len(lines):', len(lines))
        # for line in lines:
        #     print('----len(line):', len(line))

    print('len(attention_list):', len(attention_list))
    # for _item in attention_list:
    #     print('_item.shape:', _item.shape)

    total_token_num = input_token_num + generate_token_num  # 总token数
    dim_0 = attention_list[0].shape[0]
    attention_matrix = torch.zeros(dim_0,total_token_num,total_token_num)
    for i,_item in enumerate(attention_list):
        print('_item.shape:', _item.shape)
        print(i,i+_item.shape[1],i+_item.shape[2])
        attention_matrix[:,i:i+_item.shape[1],:_item.shape[2]] = _item

    # TODO: 热力图展示需要再完善
    print('attention_matrix.shape:', attention_matrix.shape)
    """展示模型inference计算生成的attention矩阵"""
    sum_matrix = torch.sum(attention_matrix,dim=0)
    plt.imshow(sum_matrix)
    plt.title('cos&sin postional embedding')
    # 添加颜色条
    cbar = plt.colorbar()
    # 添加颜色条标签
    cbar.set_label('Colorbar Label')
    plt.savefig('./pictures/sum/PositionalEmbedding')

    """对每个通道分别绘制attention热力图"""
    # for i,sub_matrix in enumerate(attention_matrix):
    #     plt.imshow(sub_matrix)
    #     plt.title('cos&sin postional embedding')
    #     # 添加颜色条
    #     cbar = plt.colorbar()
    #     # 添加颜色条标签
    #     cbar.set_label('Colorbar Label')
    #     plt.savefig(f'./pictures/temp/PositionalEmbedding_{i}')
    #     cbar.remove()

def generate(sequences_info):
    """1. 加载model和tokenizer"""
    # if torch.cuda.is_available():
    #     n_gpus = torch.cuda.device_count()
    # if is_ipex_available() and torch.xpu.is_available():
    #     n_gpus = torch.xpu.device_count()
    device_map = "auto"
    model = LlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="eager",
        load_in_4bit=True,
        # load_in_8bit=True,
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=False,
        #     load_in_8bit=True,
        #     llm_int8_threshold=6.0,
        #     llm_int8_has_fp16_weight=False,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # ),
        # token=False
        )
    
    # show_model_attribute(model)
    # model = AutoModel.from_pretrained(MODEL_PATH)
    # model = model.eval()
    """计算模型参数量"""
    # model_param_num = cal_model_params(model)

    """2. Generate"""
    diff = []  # generate和gt不同的op
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    for sequence_info in tqdm.tqdm(sequences_info,total=len(sequences_info)):
        # print(sequence_info)
        """2.1. 对输入进行编码&inference"""
        tokens = tokenizer.tokenize(sequence_info["sequence"])
        input_ids = tokenizer.encode(sequence_info["sequence"], return_tensors="pt").to(f'cuda:{args.device}')
        print('len(input_ids):', len(input_ids[0]))
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, 
                generation_config = GenerationConfig(
                    return_dict_in_generate = True,
                    output_attentions = True
                ),
                max_length=3000, 
                num_beams=3,
                )

        """2.2. 对输出进行解码,提取输出序列和attention矩阵"""
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        attention_tuple = outputs.attentions
        print('attention_tuple.shape:', len(attention_tuple))
        show_attention_matrix(attention_tuple,len(input_ids[0]),len(attention_tuple))
        # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)


        """extra: 比较generate和gt是否相同"""
        generate = generated_text.split("### Response:  ")[-1]
        gt = sequence_info["gt"]
        if generate != gt:
            # print(sequence_info["op_name"])
            diff.append(sequence_info["op_name"])
        """3. 保存generate和对应的gt"""
        if not os.path.exists(f'./evaluate/{args.model_name}-{args.ckpt}-{args.train_data_version}-{args.test_data_version}'):
            os.mkdir(f'./evaluate/{args.model_name}-{args.ckpt}-{args.train_data_version}-{args.test_data_version}')
            os.mkdir(f'./evaluate/{args.model_name}-{args.ckpt}-{args.train_data_version}-{args.test_data_version}/generate')
            os.mkdir(f'./evaluate/{args.model_name}-{args.ckpt}-{args.train_data_version}-{args.test_data_version}/gt')
            os.mkdir(f'./evaluate/{args.model_name}-{args.ckpt}-{args.train_data_version}-{args.test_data_version}/full_gen')
        # with open("./evaluate/generate/{}_generate.c".format(sequence_info["op_name"]), 'w') as file:
        # print('generate:', generate)
        with open(os.path.join(f'./evaluate/{args.model_name}-{args.ckpt}-{args.train_data_version}-{args.test_data_version}','generate',"{}_generate.c".format(sequence_info["op_name"])),'w') as file:
            file.write(generate)
        with open(os.path.join(f'./evaluate/{args.model_name}-{args.ckpt}-{args.train_data_version}-{args.test_data_version}','gt',"{}_gt.c".format(sequence_info["op_name"])),'w') as file:
            file.write(gt)
        with open(os.path.join(f'./evaluate/{args.model_name}-{args.ckpt}-{args.train_data_version}-{args.test_data_version}','full_gen',"{}_full_gen.c".format(sequence_info["op_name"])),'w') as file:
            file.write(generated_text)
    print('diff:', diff)
    return diff



if __name__ == "__main__":

    sequences_info = load_eval()  # 数据加载
    diff_op = generate(sequences_info=sequences_info)
    """数据集日志信息LOG"""
    logging.basicConfig(filename=os.path.join(f'./evaluate/{args.model_name}-{args.ckpt}-{args.train_data_version}-{args.test_data_version}','evaluate_model_info.log'), level=logging.INFO, filemode='w+')  # 日志信息
    logging.info('MODEL_PATH: %s', MODEL_PATH)
    logging.info('EVAL_DATA_PATH: %s', EVAL_DATA_PATH)
    logging.info('EVAL_PROMPT: %s', EVAL_PROMPT)
    logging.info('diff_op: %s', diff_op)

