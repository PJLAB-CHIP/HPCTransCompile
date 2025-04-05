import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import load_dataset
import argparse
import yaml
from trl import SFTConfig,SFTTrainer
import time

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

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='QwenCoder_14b.yaml', help='Path to the config file')
args = parser.parse_args()

config_dir = './config/'
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
    no_cuda=not torch.cuda.is_available()
)

def tokenize_function(examples,tokenizer):
    inputs = tokenizer(examples["prompt"], max_length=4096, truncation=True, padding="max_length")
    outputs = tokenizer(examples["completion"], max_length=4096, truncation=True, padding="max_length")
    inputs["labels"] = outputs["input_ids"]
    print(inputs)
    return inputs

def main():
    data_path = os.path.join(config.data.dataset_dir, config.data.data_name)
    model_path = config.model.model_path
    dataset = load_dataset("json",data_files=data_path,split="train")
    print('====Dataset====')
    print(dataset)
    print('=========')
    # Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir = config.model.cache_dir if config.model.cache_dir is not None else None,
        padding_side = config.model.tokenizer_pad #'left'
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir = config.model.cache_dir if config.model.cache_dir is not None else None,
        quantization_config = bnb_config if config.model.use_quant else None,
        # device_map={"": torch.cuda.current_device()},
        device_map = config.model.device_map,
        trust_remote_code = True
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    # 数据预处理
    # inputs = tokenizer(dataset["prompt"], truncation=True, padding=True)
    # labels = tokenizer(dataset["completion"], truncation=True, padding=True)
    # inputs["labels"] = labels["input_ids"]
    # print(inputs)
    # dataset = dataset.rename_column("prompt", "input_ids")
    # dataset = dataset.rename_column("completion", "labels")
    # dataset = tokenize_function(dataset["train"],tokenizer)

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset
    )

    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=dataset
    # )
    print(f"Current Device: {torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}")
    print_trainable_parameters(model)
    # 开始训练
    print("开始训练")
    start_time = time.time()
    trainer.train()

    trainer.model.save_pretrained(os.path.join(config.model.lora_save_model, config.model.model_name))
    model.config.use_cache = True
    end_time = time.time()
    print("训练结束")
    print("耗时：", end_time - start_time)

if __name__ == '__main__':
    main()
