import json
import random
import numpy as np


instruction_path = "./instruction/instruction_v1.0.json"
raw_data_path = "./raw_data/raw_data_v1.0.json"
op_attribute = []
hpc_data = []

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
    hpc_data.append({
        "instruction":instruction,
        "input":input,
        "output":output
    })

print('len(hpc_data):', len(hpc_data))

# 划分训练集&测试集
SAMPLING_RATIO = 0.95
random.shuffle(hpc_data)
train_num = (int)(len(hpc_data)*SAMPLING_RATIO)
train_dataset = hpc_data[:train_num]
test_dataset = hpc_data[train_num:]

print('len(train_dataset):', len(train_dataset))
print('len(test_dataset):', len(test_dataset))

with open('./data/hpc_data_v1.0.json', 'w') as file:
    json.dump(hpc_data, file,indent=2)

with open('./data/hpc_train_v1.0.json', 'w') as file:
    json.dump(train_dataset, file,indent=2)

with open('./data/hpc_test_v1.0.json', 'w') as file:
    json.dump(test_dataset, file,indent=2)
