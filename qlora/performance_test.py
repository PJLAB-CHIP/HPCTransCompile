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
from torch.nn import DataParallel
from torch.nn import Module
import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple, Union, Optional


def calculate_param_count():
    """
    计算模型参数量
    """
    # Embedding层
    embedding_params = 32016 * 4096

    # LlamaModel
    llama_model_params = 0
    for layer in range(32):
        llama_model_params += 4096 * 4096 * 4  # LlamaAttention
        llama_model_params += 4096 * 4096 * 2  # LlamaAttention (rotary embedding)
        llama_model_params += 4096 * 11008 * 3  # LlamaMLP

    # Linear层（lm_head）
    linear_params = 4096 * 32016

    # 总参数数量
    total_params = embedding_params + llama_model_params + linear_params

    print("Total Parameters: {:,}".format(total_params))


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    device_map = {'transformer.word_embeddings': 0,
                  'transformer.final_layernorm': 0, 'lm_head': 0}

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.layers.{i}'] = gpu_target
        used += 1

    return device_map


def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 2,
                       device_map: Optional[Dict[str, int]] = None, **kwargs) -> Module:
    if num_gpus < 2 and device_map is None:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()
    else:
        from accelerate import dispatch_model

        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half()

        if device_map is None:
            device_map = auto_configure_device_map(num_gpus)
        print('device_map:', device_map)

        model = dispatch_model(model, device_map=device_map)

    return model

# model_path = "./llama2-13b-orca-8k-3319"
# model_path = "../model/CodeLlama-7b-hf"
# model_path = "../model/PolyCoder-2.7B"
# model_path = "./save_model/CodeLlama-7b-hf-10000"
# model_path = "../model/chatglm3-6b"
# model_path = "../model/toxic-bert"
# model_path = "../model/mt5-base"
model_path = "./save_model/CodeLlama-7b-hf-1000"

model = AutoModelForCausalLM.from_pretrained(model_path)
# model = AutoModel.from_pretrained(model_path)
model = model.eval()
# print('model:', model)

"""多卡加速推理"""
# model = load_model_on_gpus(model_path,num_gpus=8)
# model = DataParallel(model)
# device_ids = [0, 1, 2, 3]


tokenizer = AutoTokenizer.from_pretrained(model_path)
# sequence = 'The following are multiple choice questions (with answers) about  abstract algebra.\n\nLet p = (1, 2, 5, 4)(2, 3) in S_5 . Find the index of <p> in S_5.\nA. 8\nB. 2\nC. 24\nD. 120\nAnswer:'
# sequence = 'Write a function to add two numpy arraies in C++\n\n### Response: '
# sequence = 'Write a cuda operator to implement matrix multiplication\n\n### Response: '
# sequence = 'Design a program that can translate a given C language operator into an equivalent CUDA operator. CUDA operators are parallel computing code written for GPUs, so consideration should be given to parallelism and GPU architecture.\n\n### Input:\nTVM_DLL int32_t default_function(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {\n    int32_t A_code = arg_type_ids[0];\n    void* A = (((TVMValue*)args)[0].v_handle);\n    void* A_1 = (((DLTensor*)A)[0].data);\n    void* default_function_A_shape = (((DLTensor*)A)[0].shape);\n    int32_t n = ((int32_t)((int64_t*)default_function_A_shape)[0]);\n    void* default_function_A_strides = (((DLTensor*)A)[0].strides);\n    int32_t stride = ((n == 1) ? 0 : ((default_function_A_strides == NULL) ? 1 : ((int32_t)((int64_t*)default_function_A_strides)[0])));\n    int32_t dev_id = (((DLTensor*)A)[0].device.device_id);\n    void* compute = TVMBackendAllocWorkspace(1, dev_id, ((uint64_t)4 * ((uint64_t)n)), 2, 32);\n    if (compute == NULL) {\n      return -1;\n    }\n    for (int32_t i0 = 0; i0 < n; ++i0) {\n      ((float*)compute)[i0] = cosf(((float*)A_1)[(i0 * stride)]);\n    }\n    if (TVMBackendFreeWorkspace(1, dev_id, compute) != 0) {\n      return -1;\n    }\n    return 0;\n  }\n\n### Response: '
# sequence = 'What is the meaning of life?\n\n### Response: '
sequence = 'Given a performance-sensitive application that includes CUDA operators, you want to run in an environment without a GPU. Please convert the CUDA operators provided to an efficient CPU implementation with as little performance penalty as possible.\n\n### Input:\n#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \\\n     (__CUDACC_VER_MAJOR__ > 11))\n#define TVM_ENABLE_L2_PREFETCH 1\n#else\n#define TVM_ENABLE_L2_PREFETCH 0\n#endif\n\n#ifdef _WIN32\n  using uint = unsigned int;\n  using uchar = unsigned char;\n  using ushort = unsigned short;\n  using int64_t = long long;\n  using uint64_t = unsigned long long;\n#else\n  #define uint unsigned int\n  #define uchar unsigned char\n  #define ushort unsigned short\n  #define int64_t long long\n  #define uint64_t unsigned long long\n#endif\nextern \"C\" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_divide, float* __restrict__ tarray);\nextern \"C\" __global__ void __launch_bounds__(1024) default_function_kernel(float* __restrict__ T_divide, float* __restrict__ tarray) {\n  if (((((int64_t)((int)blockIdx.x)) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) >> (int64_t)5)) < (int64_t)1225) {\n    T_divide[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (tarray[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] / tarray[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]);\n  }\n}\n\n### Response: '

# sequence = 'Attention is All you'
tokens = tokenizer.tokenize(sequence)
# print('Tokens:', tokens)
input_ids = tokenizer.encode(sequence, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(input_ids=input_ids, max_length=1000)

# for output in outputs:
#     print(tokenizer.decode(output))

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
