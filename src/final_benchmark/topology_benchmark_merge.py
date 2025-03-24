import os
import re
import json

# 定义目录路径
source_dir = "/code/LLM4HPCTransCompile/final_result/starcoder2-15b-instruct-v0.1/topology_expansion/pred"
source_json_file_path = "/code/LLM4HPCTransCompile/final_benchmark/generated_graph/generated_graph.json"
json_file_path = "/code/LLM4HPCTransCompile/final_benchmark/generated_graph/generated_graph.json"
# target_dir = "/mnt/data/pred"

# 创建目标目录（如果不存在）
# os.makedirs(target_dir, exist_ok=True)

# 读取JSON文件内容
with open(source_json_file_path, 'r', encoding='utf-8') as f:
    topology_data = json.load(f)

# 提取操作参数和输入形状的正则表达式模式
file_name_pattern = re.compile(r"topology_expansion_\[\['(.*?)'\]\]_\[\[(.*?)\]\].c")

# 从文件名中提取op_args和input_shape
def extract_info_from_filename(file_name):
    
    match = file_name_pattern.match(file_name)
    if match:
        op_args = match.group(1).split("', '")
        input_shape = match.group(2)
        return op_args, input_shape
    return None, None

# 遍历源目录中的所有文件
for file_name in os.listdir(source_dir):
    file_path = os.path.join(source_dir, file_name)
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 从文件名提取信息
    op_args, input_shape = extract_info_from_filename(file_name)
    
    if op_args and input_shape:
        for entry in topology_data:
            if entry["op_name"] == "topology_expansion" and entry["op_args"][0] == op_args and entry["input_shape"] == f"[[{input_shape}]]":
                # entry["deepseek_c"] = content
                entry["starcoder_c"] = content
                # entry["codellama_c"] = content


                break

# 将更新后的数据写回JSON文件
with open(json_file_path, 'w', encoding='utf-8') as f:
    json.dump(topology_data, f, ensure_ascii=False, indent=4)

print("文件内容已成功添加到字典并更新JSON文件。")
