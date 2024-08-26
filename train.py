# Imports
import os, time
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)
import torch
from datasets import load_dataset
from transformers import(
AutoTokenizer,
AutoModelForCausalLM,
BitsAndBytesConfig,
TrainingArguments,
pipeline,
logging,
TextStreamer,
Trainer
)
from peft import(
    LoraConfig, 
    PeftModel, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from finetune import DataCollatorForCausalLM
from trl import SFTTrainer
import yaml
# import wandb
import argparse

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def print_trainable_parameters(model):
    trainable_params = 0
    all_param =0
    for _, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        all_param += param.numel()
    print(f"训练参数量: {trainable_params} || 总参数量: {all_param} ||训练参数量占比%:{100*(trainable_params / all_param)}")

def generate_and_tokenize_prompt(data):
        data['length'] = len(data['input'])+len(data['output'])
        return data

parser = argparse.ArgumentParser()
parser.add_argument('--project_name', default='HPCEval finetune', type=str, help='name')
parser.add_argument('--config', type=str, default='ft_starcoder.yaml', help='Path to the config file')
args = parser.parse_args()

config_dir = './exp/'
with open(config_dir + args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = dict2namespace(config)

########################
### HPCLLM finetune ####
########################

# 加载模型和数据
model_path = os.path.join(config.model.model_hub_dir, config.model.model_name) 
data_path = os.path.join(config.data.dataset_dir, config.data.data_name) 

# 加载预训练 model&tokenizer
# 量化配置
bnb_config = BitsAndBytesConfig(
                load_in_4bit = config.bnb.q4bit,
                bnb_4bit_quant_type = config.bnb.quant_type,
                bnb_4bit_use_double_quant = config.bnb.double_quant,
                bnb_4bit_compute_dtype = torch.bfloat16
                )

# LORA & 超参配置
peft_config = LoraConfig(
            r = 8,
            lora_alpha = 16,
            lora_dropout = 0.05,
            bias = "none",
            task_type = "CAUSAL_LM",
            target_modules = ['q_proj','k_proj','v_proj'],
)
            
training_arguments = TrainingArguments(
    output_dir=os.path.join(config.training.output_dir, config.model.model_name),
    num_train_epochs=config.training.epoch,
    gradient_accumulation_steps=config.training.gradient_accumulation_steps,
    optim=config.training.optim,
    auto_find_batch_size=config.training.auto_batch,
    per_device_train_batch_size=config.training.batch_size,
    per_device_eval_batch_size=config.training.batch_size,
    gradient_checkpointing=config.training.grad_check,
    logging_steps=config.training.logging_steps,
    eval_steps=config.training.eval_steps,
    save_steps=config.training.save_steps,
    learning_rate=config.training.learning_rate,
    weight_decay=config.training.weight_decay,
    max_grad_norm=config.training.max_grad_norm,
    max_steps=config.training.max_steps,
    warmup_ratio=config.training.warmup_ratio,
    group_by_length=config.training.group_by_length,
    lr_scheduler_type=config.training.lr_scheduler_type,
    report_to=config.training.report,
)

def main():

    # 加载数据集
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(dataset["text"][0])
    # 模型加载
    model = AutoModelForCausalLM.from_pretrained(
        # model_path if config.model.cache_dir is None else "codellama/CodeLlama-13b-hf",
        model_path,
        cache_dir = config.model.cache_dir if config.model.cache_dir is not None else None,
        quantization_config = bnb_config if config.model.use_quant else None,
        device_map = config.model.device_map,
        trust_remote_code = True
    )
    # 微调前封装
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    # tokenizer加载
    tokenizer = AutoTokenizer.from_pretrained(
        # model_path if config.model.cache_dir is None else "codellama/CodeLlama-13b-hf",
        model_path,
        cache_dir = config.model.cache_dir if config.model.cache_dir is not None else None,
        padding_side = config.model.tokenizer_pad #'left'
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True

    # 模型微调
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        
    )
    print("开始训练")
    start_time = time.time()
    trainer.train()

    print_trainable_parameters(model)

    trainer.model.save_pretrained(os.path.join(config.model.lora_save_model, config.model.model_name))
    model.config.use_cache = True
    end_time = time.time()
    print("训练结束")
    print("耗时：", end_time - start_time)
    # model.eval()

if __name__ == '__main__':
    main()