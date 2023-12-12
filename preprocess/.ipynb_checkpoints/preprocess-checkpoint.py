import json
import random
import numpy as np
import os


instruction_path = "./instruction/instruction_v1.0.json"
raw_data_path = "./raw_data/raw_data_v1.0.json"
op_attribute = []
hpc_data = []
VERSION_INFO = 'v1.2'
DATASET_INFO = ""

def check_output_path(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f'create {folder_path} successfully.')
    else:
        print(f'{folder_path} already exists.')
    #

def datatset_partition(hpc_data,divide_method='shuffle',*args,**kwargs):
    if divide_method == 'shuffle':
        return dataset_partition_shuffle(hpc_data,SAMPLING_RATIO=sampling_ratio)

def dataset_partition_shuffle(hpc_data,SAMPLING_RATIO=0.95):
    SAMPLING_RATIO = 0.95
    random.shuffle(hpc_data)
    train_num = (int)(len(hpc_data)*SAMPLING_RATIO)
    train_dataset = hpc_data[:train_num]
    test_dataset = hpc_data[train_num:]
    return train_dataset,test_dataset

if __name__ == "__main__":
    with open(instruction_path, 'r') as file:
        instructions = json.load(file)

    with open(raw_data_path, 'r') as file:
        raw_datas = json.load(file)

    cuda2cpu_en = instructions["cuda2cpu_en"]

    print('instructions.keys():', instructions.keys())
    print('len(raw_datas):', len(raw_datas))

    for raw_data in raw_datas:
        input = raw_data['cuda_code']
        output = raw_data['c_code']
        instruction = random.choice(cuda2cpu_en)
        op_name = raw_data['op_name']
        hpc_data.append({
            "instruction":instruction,
            "input":input,
            "output":output,
            "op_name":op_name
        })

    print('len(hpc_data):', len(hpc_data))

    # 划分训练集&测试集
    SAMPLING_RATIO = 0.95
    # random.shuffle(hpc_data)
    # train_num = (int)(len(hpc_data)*SAMPLING_RATIO)
    # train_dataset = hpc_data[:train_num]
    # test_dataset = hpc_data[train_num:]

    train_dataset, test_dataset = datatset_partition(hpc_data,sampling_ratio=SAMPLING_RATIO)

    print('len(train_dataset):', len(train_dataset))
    print('len(test_dataset):', len(test_dataset))

    check_output_path(f'./data/hpc_{VERSION_INFO}')

    with open('./data/hpc_{}/hpc_data_{}.json'.format(VERSION_INFO,VERSION_INFO), 'w') as file:
        json.dump(hpc_data, file,indent=2)

    with open('./data/hpc_{}/hpc_train_{}.json'.format(VERSION_INFO,VERSION_INFO), 'w') as file:
        json.dump(train_dataset, file,indent=2)

    with open('./data/hpc_{}/hpc_test_{}.json'.format(VERSION_INFO,VERSION_INFO), 'w') as file:
        json.dump(test_dataset, file,indent=2)
