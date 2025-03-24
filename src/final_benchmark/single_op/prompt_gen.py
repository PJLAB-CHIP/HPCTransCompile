import os
import json

def generate_prompt_files(json_folder, prompt_template, type):

    categories = ['Elementwise', 'Reduction', 'Layout Transform', 'Logic Intensive', 'Compute Intensive']

    for category in categories:
        json_file_path = f'{json_folder}/{category}/{category}.json'
        with open(json_file_path, 'r') as f:
                operators = json.load(f)
        for op in operators:
            if op['op_name'] == 'atan':
                cuda_code_1 = op['cuda_code']
                input_shape_1 = op['input_shape']
                c_code_1 = op['c_code']
            if type == 'single':
                if op['op_name'] == 'matmul':
                    cuda_code_2 = op['cuda_code']
                    input_shape_2 = op['input_shape']
                    c_code_2 = op['c_code']
            else:
                topo_json_file_path = '/code/LLM4HPCTransCompile/final_benchmark/generated_graph/generated_graph.json'
                with open(topo_json_file_path, 'r') as f:
                    operators = json.load(f)
                    for op in operators:
                        if op['input_shape'] == '[[19, 17, 15]]':
                            cuda_code_2 = op['cuda_code']
                            input_shape_2 = op['input_shape']
                            c_code_2 = op['c_code']

    if type == 'topo':
        json_folder = '/code/LLM4HPCTransCompile/final_benchmark/generated_graph'
        json_file_path = '/code/LLM4HPCTransCompile/final_benchmark/generated_graph/generated_graph.json'
        output_folder = f'{json_folder}/prompt_few_shot'
        output_folder_c = f'{json_folder}/c'
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(output_folder_c, exist_ok=True)
        if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as f:
                    operators = json.load(f)
                # 遍历算子列表，生成prompt文件
                for op in operators:
                    # 提取算子信息
                    op_name = op['op_name']
                    op_args = op['op_args']
                    input_shape = op['input_shape']
                    output_shape = op['output_shape']
                    cuda_code = op['cuda_code']
                    c_code = op['c_code']

                    # 填充模板
                    prompt_content = prompt_template.format(
                        cuda_code_1=cuda_code_1,
                        input_shape_1=input_shape_1,
                        c_code_1=c_code_1,
                        cuda_code_2=cuda_code_2,
                        input_shape_2=input_shape_2,
                        c_code_2=c_code_2,
                        cuda_code=cuda_code,
                        input_shape=input_shape,
                    )
                    file_name = f"{op_name}_{op_args}_{input_shape}.text"
                    file_name_c = f"{op_name}_{op_args}_{input_shape}.c"

                    # 保存文件 text
                    with open(os.path.join(output_folder, file_name), 'w') as f:
                        f.write(prompt_content)
                    
                    # 保存文件 c
                    with open(os.path.join(output_folder_c, file_name_c), 'w') as f:
                        f.write(c_code)
    else:
        for category in categories:
            json_file_path = f'{json_folder}/{category}/{category}.json'
            output_folder = f'{json_folder}/{category}/prompt_few_shot'
            output_folder_c = f'{json_folder}/{category}/c'
            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(output_folder_c, exist_ok=True)
            
            # 读取每个类别的JSON文件
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as f:
                    operators = json.load(f)

                # 遍历算子列表，生成prompt文件
                for op in operators:
                    # 提取算子信息
                    op_name = op['op_name']
                    op_args = op['op_args']
                    input_shape = op['input_shape']
                    output_shape = op['output_shape']
                    cuda_code = op['cuda_code']

                    # 填充模板
                    prompt_content = prompt_template.format(
                        cuda_code_1=cuda_code_1,
                        input_shape_1=input_shape_1,
                        c_code_1=c_code_1,
                        cuda_code_2=cuda_code_2,
                        input_shape_2=input_shape_2,
                        c_code_2=c_code_2,
                        cuda_code=cuda_code,
                        input_shape=input_shape,
                    )


                    file_name = f"{op_name}_{op_args}_{input_shape}.text"
                    file_name_c = f"{op_name}_{op_args}_{input_shape}.c"

                    # 保存文件 text
                    with open(os.path.join(output_folder, file_name), 'w') as f:
                        f.write(prompt_content)
                    
                    # 保存文件 c
                    with open(os.path.join(output_folder_c, file_name_c), 'w') as f:
                        f.write(prompt_content)
                    
                    # # 保存文件 cuda
                    # with open(os.path.join(output_folder, file_name), 'w') as f:
                    #     f.write(prompt_content)


if __name__ == '__main__':

    # prompt.text的内容
    prompt_template_1 = """Task: Translate the given CUDA code to its equivalent C code.
    Context: You are provided with a CUDA code snippet that needs to be translated into C code. The translation should preserve the same functionality and structure as much as possible. Focus on translating the CUDA-specific parallel constructs into C constructs, such as using OpenMP for parallelism. The resulting C code should be complete and ready to compile.
    Following are some details of the cuda code:
    - Operation Name: {op_name}
    - Operation Arguments: {op_args}
    - Input Shape: {input_shape}
    - Output Shape: {output_shape}
    Input CUDA Code: \n{cuda_code}
    Print only single C function implementation, end with comment '|End-of-Code|'.
    """
    prompt_template_2 = """
    You are a senior CUDA and C programming expert, please translate the following CUDA Code to C Code: \n{cuda_code} 
    You must ensure that the generated C code matches the CUDA code parameter list, print only single C function implementation, end with comment '|End-of-Code|'.

    """

    prompt_template_3 = """
    {cuda_code}
    Translate the above CUDA to C code executed on CPU, the input tenor shape is {input_shape}. Print only a single C code function implementation, end with comment "|End-of-Code|".
    """
    prompt_template_pro_1 = """
    Task: Translate the given CUDA code to its equivalent High performace CPU C code.  
    Context: You are provided with a CUDA code snippet that needs to be translated into CPU C code. The translation should preserve the same functionality as much as possible. Focus on translating the CUDA-specific parallel constructs into the constructs supported by CPU, such as using OpenMP for parallelism. The resulting CPU C code should be complete and ready to compile. 
    Input CUDA Code: {cuda_code}
    Ihe input tensor shape is {input_shape} respectively, Print only single C function implementation, end with comment '|End-of-Code|'.
    """

    prompt_template_pro_2 = """
    {cuda_code}
    Translate the above CUDA code to CPU C code, the input tensor shape is {input_shape} respectively. Print only single C code function implementation, end with comment "|End-of-Code|".
    """

    prompt_few_shot = """
    Task: Translate the given CUDA code to its equivalent high-performance CPU C code.
    Context: You are provided with a CUDA code snippet that needs to be translated into CPU C code. The translation should preserve the same functionality as much as possible. Focus on translating the CUDA-specific parallel constructs into constructs supported by the CPU, such as using OpenMP for parallelism. The resulting CPU C code should be complete and ready to compile.

    Example 1:
    Input CUDA Code: {cuda_code_1}
    Input Tensor Shape: {input_shape_1}
    Output C Code: {c_code_1}
    //|End-of-Code|

    Example 2:
    Input CUDA Code: {cuda_code_2}
    Input Tensor Shape: {input_shape_2}
    Output C Code: {c_code_2}
    //|End-of-Code|

    Now translate the following CUDA code to its equivalent high-performance CPU C code:
    Input CUDA Code: {cuda_code}
    Input Tensor Shape: {input_shape}
    Print only a single C function implementation, ending with the comment '|End-of-Code|'.
    """

    # 调用函数生成prompt文件
    json_file_path = '/code/LLM4HPCTransCompile/final_benchmark/single_op'  # 请替换为实际的JSON文件路径
    gen_type = 'topo'
    # gen_type = 'single'
    generate_prompt_files(json_file_path, prompt_few_shot, gen_type)
