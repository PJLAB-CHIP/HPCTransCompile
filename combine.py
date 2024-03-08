import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, CodeLlamaTokenizer
import argparse
import logging
import os


def save_model(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    base_tokenizer = CodeLlamaTokenizer.from_pretrained(model_name_or_path)

    print(f"Loading the LoRA adapter from {lora_path}")
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    """命令行解析参数信息"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,help='foundation model')
    parser.add_argument('--data_version',type=str)
    parser.add_argument('--ckpt',type=int)
    args = parser.parse_args()  # 解析命令行参数

    foundation_model_path = os.path.join('../../model',args.model_name)
    save_model_path = os.path.join('./save_model',f'{args.model_name}-{args.ckpt}-{args.data_version}')
    lora_path = os.path.join('./output',f'hpc_{args.data_version}',f'checkpoint-{args.ckpt}')
    # lora_path = os.path.join('./output',f'checkpoint-{args.ckpt}')

    save_model(foundation_model_path,output_path=save_model_path,lora_path=lora_path)
    """数据集日志信息LOG"""
    logging.basicConfig(filename=os.path.join(save_model_path,'save_model_info.log'), level=logging.INFO, filemode='w+')  # 日志信息
    logging.info('Foundation model: %s', foundation_model_path)
    logging.info('lora_path: %s', lora_path)
    logging.info('ckpt: %s', args.ckpt)
    logging.info('data_version: %s', args.data_version)

