import os
import json
import random
import shutil

def single_benchmark(json_file_path, source_folder):
    # 读取JSON文件
    with open(json_file_path, 'r') as f:
        operators = json.load(f)

    # 定义算子类别及对应的算子名称
    categories = { 'Elementwise':['erf', 'leaky_relu', 'sqrt', 'asin', 'tanh', 'floor', 'log', 'sin', 'prelu', 'fast_exp', 'log2', 'sign', 'sigmoid', 'isnan', 'relu', 'cos', 'fast_tanh', 'log10', 'round', 'atan', 'negative', 'add', 'tan', 'atanh', 'acos', 'ceil', 'asinh', 'abs', 'exp','rsqrt', 'sinh', 'cosh', 'combination_op', 'fast_erf'],
                     'Reduction':['min', 'global_pool_max', 'global_pool_avg', 'sum', 'max', 'rms_norm', 'log_softmax', 'softmax', 'fast_softmax', 'softmax_common', 'prod', 'pool3d', 'pool1d', 'pool2d', 'adaptive_pool_max', 'adaptive_pool_avg'], 
                     'Layout Transform':['reshape', 'transpose', 'gather_nd', 'scatter_nd', 'reorg','unpack_NCHWc_to_nchw', 'flatten', 'scale_shift_nchw', 'flip', 'depth_to_space', 'batch_to_space_nd', 'strided_slice', 'space_to_depth', 'scale_shift_nchwc', 'mirror_pad', 'dilate'], 
                     'Logic Intensive':['fifo_buffer', 'multi_out_op', 'shape', 'upsampling','resize2d', 'resize3d', 'grid_sample', 'argsort'], 
                     'Compute Intensive':['lrn', 'matmul','conv2d_opt', 'dft', 'group_conv2d_opt', 'batch_matmul_opt'] 
                     }
    
    base_dir = '/code/LLM4HPCTransCompile/final_benchmark/single_op'
    # 为每个类别创建文件夹
    for category in categories:
        os.makedirs(f'/code/LLM4HPCTransCompile/final_benchmark/single_op/{category}', exist_ok=True)

    # 记录已处理的算子
    processed_ops = set()

    # 创建各类别的列表
    selected_ops = {
        'Elementwise': [],
        'Reduction': [],
        'Layout Transform': [],
        'Logic Intensive': [],
        'Compute Intensive': []
    }
    
    single_ops = []
    # 遍历算子列表
    for op in operators:
        op_name = op['op_name']
        if op_name in processed_ops:
            continue
        
        # 找到算子对应的类别
        category = None
        for cat, ops in categories.items():
            if op_name in ops:
                category = cat
                break

        if category:
            # 随机选择一个算子
            selected_op = random.choice([o for o in operators if o['op_name'] == op_name])
            selected_ops[category].append(selected_op)
            # single_ops.append(selected_op)
            # # 生成文件名
            # # json_name = f"{category}.json"
            # file_name = f"{selected_op['op_name']}_{selected_op['op_args']}_{selected_op['input_shape']}"

            #  # 遍历源文件夹中的子文件夹
            # for subfolder in ['c', 'cuda', 'device_module', 'host_module', 'ir']:
            #     source_subfolder = os.path.join(source_folder, subfolder)
            #     if os.path.exists(source_subfolder):
            #         for src_file in os.listdir(source_subfolder):
            #             if src_file.startswith(file_name):
            #                 # 构造源文件路径
            #                 source_file_path = os.path.join(source_subfolder, src_file)
                            
            #                 # 构造目标文件路径
            #                 target_file_path = os.path.join(f'{base_dir}/{category}/{subfolder}', src_file)
            #                 os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
                            
            #                 # 复制文件
            #                 shutil.copy2(source_file_path, target_file_path)
            
            processed_ops.add(op_name)
            # 保存算子信息到文件
    for category, ops_list in selected_ops.items():
        print(category)
        with open(f'{base_dir}/{category}/{category}.json', 'w') as f:
            json.dump(ops_list, f, indent=4)
    # with open(f'{base_dir}/single.json', 'w') as f:
    #         json.dump(single_ops, f, indent=4)

if __name__ == '__main__':
    singe_op_path = "/code/LLM4HPCTransCompile/final_benchmark/single_op/single_op.json"
    source_folder = "/code/LLM4HPCTransCompile/data/single"
    single_benchmark(singe_op_path, source_folder)