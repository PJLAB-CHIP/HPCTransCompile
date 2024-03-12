from ast import arg
from curses import meta
import logging
import argparse
import os
import sys
from dataclasses import dataclass,field
from time import sleep
from unittest import result
from git import Optional, Union
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    Seq2SeqTrainer,
    CodeLlamaTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    get_peft_model_state_dict
)
import transformers
from datasets import load_dataset
from os.path import (
    isdir,
    join,
    exists
)
from utils import (
    show_model_info,
    extract_alpaca_dataset
)
from args import(
    ModelArguments,
    DataArguments,
    TrainingArguments
)
from typing import Optional,Sequence,Dict
import torch
from torch.nn.utils.rnn import pad_sequence
import copy
import subprocess
from torch import optim

# os.environ['WANDB_API_KEY'] = 'cfb5ba8f1bb02b39b518c24874b8579617459db3'
# os.environ['WANDB_MODE'] = 'offline'

IGNORE_INDEX = -100

# TODO: 进一步完善DataCollator模块
@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
                tokenized_sources_with_prompt['input_ids'],
                tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True,
                              padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict



def get_data_module(args):
    def format_dataset(dataset):
        """根据数据格式对input和output进行填充形成完整的prompt"""
        if args.dataset_format == 'alpaca' or args.dataset_format == 'alpaca-clean' or (args.dataset_format is None and args.dataset_name_or_path in ['alpaca','alpaca-clean']):
            dataset = dataset.map(extract_alpaca_dataset,remove_columns=['instruction','op_name'])
        elif args.dataset_format == 'input-output':
            pass
        return dataset
    
    def generate_and_tokenize_prompt(data):
        data['length'] = len(data['input'])+len(data['output'])
        return data
    
    if args.dataset_name_or_path.endswith('.json') or args.dataset_name_or_path.endswith('.jsonl'):
        data = load_dataset('json',data_files=args.dataset_name_or_path)
    else:
        data = load_dataset(args.dataset_name_or_path)
    
    data = format_dataset(data)
        
    # train|val|test划分
    data = data['train'].train_test_split(train_size=args.train_size,seed=42)
    print('data[\'train\']:', data['train'])
    print('data[\'test\']:', data['test'])
    train_data = data['train'].shuffle().map(generate_and_tokenize_prompt)
    val_data = data['test'].shuffle().map(generate_and_tokenize_prompt)
    return train_data,val_data

def get_accelerate_model(args):
    if not args.full_finetune:  
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path = args.model_name_or_path,
            device_map=args.device_map,
            load_in_4bit = (not args.full_finetune) and args.bits == 4,
            load_in_8bit = (not args.full_finetune) and args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit = (not args.full_finetune) and args.bits == 4,
                load_in_8bit = (not args.full_finetune) and args.bits == 8,
                bnb_4bit_quant_type=args.quant_type,
                bnb_4bit_use_double_quant=args.double_quant,
                bnb_4bit_compute_dtype=torch.bfloat16
            ),
        ) 
    else:
        model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path = args.model_name_or_path,
        device_map=args.device_map,
        )
    
    if args.use_lora:
        model = prepare_model_for_kbit_training(model) # wraps the entire protocol for preparing a model before running a training.
        config = LoraConfig(
            r = 8,
            lora_alpha = 16,
            target_modules = ['q_proj','k_proj','v_proj'],
            lora_dropout = 0.05,
            bias = "none",
            task_type = "CAUSAL_LM",
        ) # target_modules控制在哪些层加LORA
        model = get_peft_model(model,config) # LlamaForCausalLM --> PeftModelForCausalLM


    checkpoint_dir,completed_training = get_last_checkpoint(args.output_dir)
    print('checkpoint_dir:', checkpoint_dir)
    # TODO: 从checkpoint加载模型
    if not completed_training:
        pass
    tokenizer = CodeLlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path = args.model_name_or_path,
        padding_side = 'left'
    )
    print('vars(tokenizer):', vars(tokenizer))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.max_seq_len = args.max_seq_len
    return model,tokenizer

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir,'completed'))
        if is_completed:
            return None,True
        max_step = 0
        for folder_name in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir,folder_name)) and folder_name.startswith('checkpoint'):
                max_step = max(max_step,int(folder_name.replace('checkpoint-','')))
        if max_step == 0:
            return None, is_completed
        checkpoint_dir = join(checkpoint_dir,f'checkpoint-{max_step}')
        print(f'Find a previous checkpoint at: {checkpoint_dir}')
        return checkpoint_dir,is_completed
    else:
        raise ValueError(f'Expected checkpoint_dir be a dir, but get {checkpoint_dir}')

def train():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments,DataArguments,TrainingArguments)
    )
    model_args,data_args,training_args,extra_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

    args = argparse.Namespace(
        **vars(model_args),**vars(data_args),**vars(training_args)
    )
    print('args:', args)
    if not exists(path=args.output_dir):
        os.mkdir(args.output_dir)
    model, tokenizer = get_accelerate_model(args)
    train_data,val_data = get_data_module(args)
    # print('train_data:', train_data)
    # print('val_data:', val_data)
    # optimizer = optim.SGD(model.parameters(),lr=args.learning_rate)
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            output_dir=args.output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            group_by_length=args.group_by_length,
            report_to="wandb" if args.use_wandb else None,
            run_name=args.wandb_run_name if args.use_wandb else None,
        ),
        data_collator = DataCollatorForCausalLM(
            tokenizer=tokenizer,
            source_max_len=args.source_max_len,
            target_max_len=args.target_max_len,
            train_on_source=False,
            predict_with_generate=False,
        ),
    )
    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    trainer.train()
    model.save_pretrained(args.output_dir)

if __name__ == '__main__':
    train()