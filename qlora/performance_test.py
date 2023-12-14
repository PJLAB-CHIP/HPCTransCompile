from re import split
from sympy import sequence
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    AutoModel,
    pipeline
)
import torch
import os
import json
import tqdm
import argparse
import logging

"""命令行解析参数信息"""
parser = argparse.ArgumentParser()
parser.add_argument('--model_name',type=str,help='foundation model')
parser.add_argument('--train_data_version',type=str)
parser.add_argument('--test_data_version',type=str)
parser.add_argument('--ckpt',type=int)
parser.add_argument('--device',type=int,default=0)
parser.add_argument('--use_lora_model',type=bool,default=False)
args = parser.parse_args()  # 解析命令行参数

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

def show_model_attribute(model):
    """
    显示模型属性
    """
    # print('model:', model)
    """
    LlamaForCausalLM(
    (model): LlamaModel(
        (embed_tokens): Embedding(32016, 5120)
        (layers): ModuleList(
        (0-39): 40 x LlamaDecoderLayer(
            (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=5120, out_features=5120, bias=False)
            (k_proj): Linear(in_features=5120, out_features=5120, bias=False)
            (v_proj): Linear(in_features=5120, out_features=5120, bias=False)
            (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=5120, out_features=13824, bias=False)
            (up_proj): Linear(in_features=5120, out_features=13824, bias=False)
            (down_proj): Linear(in_features=13824, out_features=5120, bias=False)
            (act_fn): SiLUActivation()
            )
            (input_layernorm): LlamaRMSNorm()
            (post_attention_layernorm): LlamaRMSNorm()
        )
        )
        (norm): LlamaRMSNorm()
    )
    (lm_head): Linear(in_features=5120, out_features=32016, bias=False)
    )
    """
    # print('model.state_dict().keys():', model.state_dict().keys())
    print('model parameter matrices num:', len(model.state_dict().keys()))
    """
    1. embedding层
    'model.embed_tokens.weight' torch.Size([32016, 5120])
    2. transformer层 0-39
    'model.layers.0.self_attn.q_proj.weight', torch.Size([5120, 5120])
    'model.layers.0.self_attn.k_proj.weight', torch.Size([5120, 5120])
    'model.layers.0.self_attn.v_proj.weight', torch.Size([5120, 5120])
    'model.layers.0.self_attn.o_proj.weight', torch.Size([5120, 5120])
    'model.layers.0.mlp.gate_proj.weight', torch.Size([13824, 5120])
    'model.layers.0.mlp.up_proj.weight', torch.Size([13824, 5120])
    'model.layers.0.mlp.down_proj.weight', torch.Size([5120, 13824])
    'model.layers.0.input_layernorm.weight', torch.Size([5120])
    'model.layers.0.post_attention_layernorm.weight' torch.Size([5120])
    3. norm层
    'model.norm.weight', torch.Size([5120])
    'lm_head.weight' torch.Size([32016, 5120])
    """

def generate(sequences_info):
    """1. 加载model和tokenizer"""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"":args.device}
        )
    # show_model_attribute(model)
    # model = AutoModel.from_pretrained(MODEL_PATH)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    """2. Generate"""
    diff = []  # generate和gt不同的op
    for sequence_info in tqdm.tqdm(sequences_info,total=len(sequences_info)):
        # print(sequence_info)
        tokens = tokenizer.tokenize(sequence_info["sequence"])
        input_ids = tokenizer.encode(sequence_info["sequence"], return_tensors="pt").to(f'cuda:{args.device}')
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, max_length=4000, num_beams=4)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
    logging.basicConfig(filename=os.path.join(f'./evaluate/{args.model_name}-{args.train_data_version}-{args.test_data_version}','evaluate_model_info.log'), level=logging.INFO, filemode='w+')  # 日志信息
    logging.info('MODEL_PATH: %s', MODEL_PATH)
    logging.info('EVAL_DATA_PATH: %s', EVAL_DATA_PATH)
    logging.info('EVAL_PROMPT: %s', EVAL_PROMPT)
    logging.info('diff_op: %s', diff_op)

