import json
import random


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

with open('./data/hpc_data_v1.0.json', 'w') as file:
    json.dump(hpc_data, file,indent=2)
