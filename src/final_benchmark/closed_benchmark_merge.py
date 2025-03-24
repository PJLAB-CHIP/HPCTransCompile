import os
import re
import json

single_ops_type = ['Elementwise', 'Layout Transform', 'Reduction', 'Logic Intensive', 'Compute Intensive']
# 定义目录路径
# source_dir = "/code/LLM4HPCTransCompile/result/starcoder2-15b-instruct-v0.1/single/Reduction/pred"
# source_json_file_path = "/code/LLM4HPCTransCompile/benchmark/single_op/Reduction/Reduction.json"
# json_file_path = "/code/LLM4HPCTransCompile/benchmark/single_op/Reduction/Reduction_starcoder.json"


# 提取操作参数和输入形状的正则表达式模式

# 从文件名中提取op_args和input_shape
def extract_info_from_filename(file_name):
    pattern = re.compile(r"(.*?)_\[(.*?)\]_\[(.*?)\]\.c")
    match = pattern.match(file_name)
    if match:
        op_name = match.group(1)
        op_args = [int(x) for x in match.group(2).split(", ")]
        input_shape = f"[[{match.group(3)}]]"
        return op_name, op_args, input_shape
    return None, None, None

# model_full_name = "CodeLlama-13b-hf"
model_full_name = "DeepSeek-Coder-V2-Lite-Instruct"
# model_full_name = "starcoder2-15b-instruct-v0.1"

# model_name = model_full_name.split('-')[0]
model_name = "deepseek"

# closed_list = ['cluade1_c', 'cluade2_c', 'gemini1_c', 'gemini2_c', 'gpt1_c', 'gpt2_c']
closed_list = ['gemini3_c']


for closed in closed_list:

    # source_dir = os.path.join('/code/LLM4HPCTransCompile/final_result/', f'{model_full_name}/single/{type}/pred')

    source_dir = os.path.join('/code/LLM4HPCTransCompile/closed_test_3', f'{closed}/single_op')
    # source_dir = os.path.join('/code/LLM4HPCTransCompile/closed_test_3', f'{closed}/topology')


    source_json_file_path = os.path.join('/code/LLM4HPCTransCompile/benchmark/single_op', f'{type}/{type}.json')
    # source_json_file_path = '/code/LLM4HPCTransCompile/final_benchmark/single_op/closed_test.json'

    # source_json_file_path = "/code/LLM4HPCTransCompile/benchmark/single_op/Reduction/Reduction.json"
    # json_file_path = "/code/LLM4HPCTransCompile/benchmark/single_op/Reduction/Reduction_starcoder.json"
    # save_json = type+'-'+model_name
    # json_file_path = os.path.join('/code/LLM4HPCTransCompile/final_benchmark/single_op', f'{type}.json')

    json_file_path = '/code/LLM4HPCTransCompile/final_benchmark/single_op/single_op.json'
    # json_file_path = '/code/LLM4HPCTransCompile/final_benchmark/generated_graph/generated_graph.json'



    # 读取JSON文件内容

    with open(json_file_path, 'r', encoding='utf-8') as f:
        topology_data = json.load(f)

    # 遍历源目录中的所有文件
    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 从文件名提取信息
        y = file_name.split('_')
        input_shape = y[-1].split('.')[0]
        print(input_shape)
        # op_name, op_args, input_shape = extract_info_from_filename(file_name)
        # print(op_name, op_args, input_shape)
        
        # if op_args and input_shape:
        for entry in topology_data:
            # if entry["op_name"] == op_name and entry["op_args"][0] == op_args and entry["input_shape"] == f"[[{input_shape}]]":
            if entry["input_shape"] == input_shape:
                # entry["codellama_c"] = content
                # entry["starcoder_c"] = content
                entry[closed] = content

                print("op add", y[0])
                break

    # 将更新后的数据写回JSON文件
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(topology_data, f, ensure_ascii=False, indent=4)

    print("文件内容已成功添加到字典并更新JSON文件。")
