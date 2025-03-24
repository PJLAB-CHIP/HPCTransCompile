import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
import numpy as np
import logging

logging.basicConfig(
    filename='log.log',  # 输出到文件
    filemode='w',
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s %(levelname)s: %(message)s',  # 设置日志格式
    datefmt='%Y-%m-%d %H:%M:%S'  # 设置日期格式
)

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
    return cosine_similarity

def extract_embedding_output(model,input_ids):
    decoder_layers = model.model.layers
    embedding_output = None
    for layer in decoder_layers:
        logging.info(layer)
        output = layer(input_ids)
        logging.info(output.shape)
        if embedding_output is None:
            embedding_output = output
        else:
            embedding_output = torch.cat([embedding_output, output], dim=1)
    return embedding_output

def calculate_similarity(model,tokenizer,c_code,cuda_code):
    logging.info(model)
    c_input = tokenizer.encode(c_code,return_tensors="pt")
    logging.info(c_input)
    c_embeddings = extract_embedding_output(model,c_input['input_ids'])
    logging.info(c_embeddings)
    c_output = model(**c_input)
    logging.info(c_output)
    # c_embeddings = c_output.last_hidden_state[:, 0, :]

    cuda_input = tokenizer(cuda_code,return_tensors="pt")
    cuda_output = model(**cuda_input)
    cuda_embeddings = cuda_output.last_hidden_state[:, 0, :]

    similarity = cosine_similarity(c_embeddings.detach().numpy(), cuda_embeddings.detach().numpy())
    return similarity


if __name__ == '__main__':
    model_path = '/code/LLM4HPCTransCompile/model/DeepSeek-Coder-V2-Lite-Instruct'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map = 'auto',
        trust_remote_code = True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
        )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    c_code = "hello world"
    cuda_code = "你好"

    similarity = calculate_similarity(model,tokenizer,c_code,cuda_code)
    print('similarity:', similarity)