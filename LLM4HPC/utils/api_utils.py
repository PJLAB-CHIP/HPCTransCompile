import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

class APIConfig:
    # DeepSeek
    deepseek_key = 'sk-7809ba52a405441592d73b093a9b0ea4'
    deepseek_url = 'https://api.deepseek.com'
    deepseek_config = {
        "max_tokens": 2000,
        "temperature":0.7
    }

class MODEL_PATH:
    CodeLlama_34b = '/code/model_hub/models--meta-llama--CodeLlama-34b-hf/snapshots/d5270a637cc64c394ad5369880f511c6440dd4cd'
    Llama2_13b = '/code/model_hub/models--meta-llama--llama-2-13b-hf/snapshots/5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1'
    QwenCoder_14b = '/code/model_hub/models--Qwen--Qwen2.5-Coder-14B-Instruct/snapshots/7fc9a75b07a3f8f325985b8a10af2e73d7cd63c3'
    Qwen_14b = '/code/model_hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8'

def local_call(model_name,system_prompt,base_prompt,device,is_save=True):
    # Model & Tokenizer
    if model_name == 'QwenCoder_14b':
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH.QwenCoder_14b,
            trust_remote_code = True,
            torch_dtype=torch.float16,
            load_in_4bit=True
        ).to(device)
        # model = AutoModelForCausalLM.from_pretrained(
        #     MODEL_PATH.QwenCoder_14b,
        #     device_map="auto",
        #     trust_remote_code = True,
        #     torch_dtype=torch.float16,
        #     load_in_8bit=True
        # )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH.QwenCoder_14b)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
    else:
        raise ValueError(f'Unsupport Model: {model_name}.')
    # Generate
    prompt = f'{system_prompt}\n\n{base_prompt}'
    inputs = tokenizer(prompt,return_tensors='pt').to(device)
    # outputs = model.generate(
    #     inputs["input_ids"],
    #     max_length=8192,  # 设置最大生成长度
    #     do_sample=False   # 是否启用采样
    # )
    outputs = model.generate(
        inputs["input_ids"],
        max_length=16384,  # 设置最大生成长度
        temperature=0.1, # 控制生成随机性
        top_p=0.1,       # 调整采样范围
        do_sample=True   # 是否启用采样
    )
    generated_content = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pattern = re.escape(prompt)
    generated_content = re.sub(f'^{pattern}', '', generated_content, count=1).strip()
    # print('====generated_content====')
    # print(generated_content)
    # print('========')
    return generated_content


def api_call(model_name,system_prompt,base_prompt,is_save=True):
    if model_name == 'deepseek':
        client = openai.OpenAI(
            api_key=APIConfig.deepseek_key,
            base_url=APIConfig.deepseek_url
        )
        messages = [
            {"role":"system","content":system_prompt},
            {"role":"user","content":base_prompt}
        ]
        response = client.chat.completions.create(
            model="deepseek-chat",  # 指定模型，可以是 "deepseek-chat"（V3） 或 "deepseek-reasoner"（R1）
            messages=messages,
            max_tokens=APIConfig.deepseek_config["max_tokens"],  # 控制输出最大长度
            temperature=APIConfig.deepseek_config["temperature"]  # 控制输出的随机性
            )
        return response.choices[0].message.content
        # return response


if __name__ == '__main__':
    system_prompt = "你是一个乐于助人的AI助手。"
    base_prompt = "请用简洁的语言解释量子计算的基本概念。"
    api_call('deepseek',system_prompt,base_prompt)