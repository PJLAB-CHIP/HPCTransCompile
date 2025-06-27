import json


PROMPT = """You are a software engineer tasked with translating CUDA kernel code into C++ CPU code.

Translate the CUDA kernel code (in "```cuda" tags) into C++ CPU code.

<instructions> 
- Write C++ code that performs the **exact same operation** as the CUDA kernel code. 
- Ensure that the CPU implementation preserves all numerical precision and functionality. 
- The code must be written in a multi-threaded manner using **OpenMP** or another CPU parallelism library, if appropriate. 
- Pay attention to the differences in data type definitions between CUDA and CPU platforms.
- Return the code in "```cpu" tags. 
</instructions>

```cuda
{CUDA_CODE}
```

Now, translate the CUDA kernel code into C++ CPU code."""

COMPLETION = """```cpp
{CPU_CODE}
```
"""


def load_multiple_json(file_path):
    """
    从文件中加载多个 JSON 对象，支持逐行或紧密拼接的 JSON 结构。
    
    :param file_path: JSON 文件路径
    :return: 解析出的 JSON 对象列表
    """
    json_objects = []
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read().strip()
        # 尝试直接解析为 JSON 数组
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            pass
        # 尝试按行解析（适用于每行是一个独立 JSON 对象的情况）
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        except json.JSONDecodeError:
            pass
        # 尝试按分隔符拆分（适用于 "}{" 直接拼接的情况）
        try:
            modified_data = "[" + data.replace("}{", "},{") + "]"
            return json.loads(modified_data)
        except json.JSONDecodeError:
            raise ValueError("无法解析 JSON 文件，请检查文件格式。")
    return json_objects

def load_paired_cuda_cpu(file_path_list):
    paired_cuda_cpu_list = []
    for file_path in file_path_list:
        json_objects = load_multiple_json(file_path)
        paired_cuda_cpu_list.extend(json_objects)
    print(f'Paired CUDA Kernel And CPU C++ Code Num: {len(paired_cuda_cpu_list)}.')
    return paired_cuda_cpu_list

def save_paired_cuda_cpu(file_path_list,save_path):
    # jsonl
    data_json = []
    paired_cuda_cpu_list = load_paired_cuda_cpu(file_path_list)
    for paired_cuda_cpu in paired_cuda_cpu_list:
        prompt = PROMPT.format(CUDA_CODE=paired_cuda_cpu['cuda_code'])
        completion = COMPLETION.format(CPU_CODE=paired_cuda_cpu['c_code'])
        data_json.append(
            {
                "prompt":prompt,
                "completion":completion
            }
        )
    with open(save_path,'w') as file:
        for item in data_json:
            json.dump(item, file, ensure_ascii=False)
            file.write("\n")  # 每个字典保存成一行
        # json.dump(data_json,file,ensure_ascii=False)
    # json
    # data_json = {
    #     "prompt":[],
    #     "completion":[]
    # }
    # paired_cuda_cpu_list = load_paired_cuda_cpu(file_path_list)
    # for paired_cuda_cpu in paired_cuda_cpu_list:
    #     prompt = PROMPT.format(CUDA_CODE=paired_cuda_cpu['cuda_code'])
    #     completion = COMPLETION.format(CPU_CODE=paired_cuda_cpu['c_code'])
    #     data_json["prompt"].append(prompt)
    #     data_json["completion"].append(completion)
    # with open(save_path,'w') as file:
    #     json.dump(data_json,file,ensure_ascii=False,indent=4)


if __name__ == '__main__':
    # json_objects = load_multiple_json('/code/LLM4HPCTransCompile/TrainEngine/dataset/single_v1.0/generation.json')
    # json_objects = load_multiple_json('/code/LLM4HPCTransCompile/TrainEngine/dataset/topology_v1.0/generation.json')
    # print(len(json_objects))
    # load_paired_cuda_cpu([
    #     '/code/LLM4HPCTransCompile/TrainEngine/dataset/single_v1.0/generation.json',
    #     '/code/LLM4HPCTransCompile/TrainEngine/dataset/topology_v1.0/generation.json'
    # ])
    save_paired_cuda_cpu([
        '/code/LLM4HPCTransCompile/TrainEngine/dataset/single_v1.0/generation.json',
        '/code/LLM4HPCTransCompile/TrainEngine/dataset/topology_v1.0/generation.json'
    ],'/code/LLM4HPCTransCompile/TrainEngine/dataset/data.jsonl')