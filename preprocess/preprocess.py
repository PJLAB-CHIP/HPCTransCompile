from ast import arg, parse
import json
import random
import numpy as np
import os
import logging
import argparse
from utils import CONFIG

hpc_data = []

def check_output_path(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f'create {folder_path} successfully.')
    else:
        print(f'{folder_path} already exists.')

def dataset_partition(hpc_data,divide_method='shuffle',sampling_ratio=None):
    """
    数据集划分
    """
    if divide_method == 'shuffle':
        return dataset_partition_shuffle(hpc_data,SAMPLING_RATIO=sampling_ratio)
    elif divide_method == 'by_name':
        return dataset_partition_by_name(hpc_data,SAMPLING_RATIO=sampling_ratio)

def dataset_partition_shuffle(hpc_data,SAMPLING_RATIO=0.95):
    random.shuffle(hpc_data)
    train_num = (int)(len(hpc_data)*SAMPLING_RATIO)
    train_dataset = hpc_data[:train_num]
    test_dataset = hpc_data[train_num:]
    return train_dataset,test_dataset

def dataset_partition_by_name(hpc_data,SAMPLING_RATIO=0.95):
    # TODO: 按算子划分训练
    op_dict = {}
    for op in hpc_data:
        if op['op_name'] in op_dict:
            op_dict[op['op_name']].append(op)
        else:
            op_dict[op['op_name']] = []
            op_dict[op['op_name']].append(op)
    train_num = (int)(len(op_dict.keys())*SAMPLING_RATIO)  # 训练集算子数
    op_name_list = list(op_dict.keys())

    # 设置随机种子
    random_seed = 42
    random.seed(random_seed)
    random.shuffle(op_name_list)
    # print('op_name:', op_name_list)
    # print('op_dict.keys():', op_dict.keys())
    train_dataset_op = op_name_list[:train_num]
    test_dataset_op = op_name_list[train_num:]
    logging.info('train_dataset_op: %s', train_dataset_op)
    logging.info('test_dataset_op: %s', test_dataset_op)
    train_dataset = []
    test_dataset = []
    for train_op in train_dataset_op:
        train_dataset.extend(op_dict[train_op])
    for test_op in test_dataset_op:
        test_dataset.extend(op_dict[test_op])
    return train_dataset,test_dataset

if __name__ == "__main__":
    """命令行解析参数信息"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_ir',type=bool,default=False)
    parser.add_argument('--SAMPLING_RATIO',type=float,default=0.95)
    parser.add_argument('--VERSION_INFO',type=str)
    parser.add_argument('--raw_data_path', type=str)
    parser.add_argument('--describe')
    parser.add_argument('--use_alpaca',type=bool)
    parser.add_argument('--alpaca_path',type=str)
    args = parser.parse_args()  # 解析命令行参数
    print('args:', args)
    VERSION_INFO = args.VERSION_INFO
    raw_data_path = CONFIG.raw_data_path

    """数据集日志信息LOG"""
    check_output_path(f'./data/hpc_{VERSION_INFO}')
    logging.basicConfig(filename=os.path.join(f'./data/hpc_{VERSION_INFO}','info.log'), level=logging.INFO, filemode='w+')  # 日志信息
    logging.info('VERSION_INFO: %s', VERSION_INFO)
    logging.info('instruction_path: %s', CONFIG.instruction_path)
    logging.info('raw_data_path: %s', raw_data_path)
    logging.info('Describe: %s', args.describe)

    """加载instruction"""
    with open(CONFIG.instruction_path, 'r') as file:
        instructions = json.load(file)

    """加载原始数据集"""
    if isinstance(raw_data_path,str):
        with open(raw_data_path, 'r') as file:
            raw_datas = json.load(file)
    elif isinstance(raw_data_path,list):
        raw_datas = []
        for _path in raw_data_path:
            with open(_path, 'r') as file:
                raw_datas.extend(json.load(file))

    cuda2cpu_en = instructions["cuda2cpu_en"]
    # args.use_ir = False
    print('args.use_ir:', args.use_ir)

    """为每条raw data生成对应的prompt"""
    for raw_data in raw_datas:
        instruction = random.choice(cuda2cpu_en)
        """是否使用IR信息"""
        if not args.use_ir:
            input = raw_data['cuda_code']
        else:
            input = CONFIG.WITH_IR_INPUT_TEMPLATE.format(raw_data['ir_code'],raw_data['cuda_code'])
            ir_info = random.choice(instructions["ir_info_en"])
            instruction += ir_info
        
        output = raw_data['c_code']
        op_name = raw_data['op_name']
        hpc_data.append({
            "instruction":instruction,
            "input":input,
            "output":output,
            "op_name":op_name
        })

    """[optional] 加入alpaca数据集"""
    with open(args.alpaca_path,'r') as file:
        alpaca_data = json.load(file)
        for _item in alpaca_data:
            hpc_data.append({
                "instruction":_item['instruction'],
                "input":_item['input'],
                "output":_item['output'],
                "op_name":'alpaca'
            })

    print('len(hpc_data):', len(hpc_data))

    # 划分训练集&测试集
    train_dataset, test_dataset = dataset_partition(hpc_data,sampling_ratio=args.SAMPLING_RATIO,divide_method='by_name')

    print('len(train_dataset):', len(train_dataset))
    print('len(test_dataset):', len(test_dataset))


    with open('./data/hpc_{}/hpc_data_{}.json'.format(VERSION_INFO,VERSION_INFO), 'w') as file:
        json.dump(hpc_data, file,indent=2)

    with open('./data/hpc_{}/hpc_train_{}.json'.format(VERSION_INFO,VERSION_INFO), 'w') as file:
        json.dump(train_dataset, file,indent=2)

    with open('./data/hpc_{}/hpc_test_{}.json'.format(VERSION_INFO,VERSION_INFO), 'w') as file:
        json.dump(test_dataset, file,indent=2)
