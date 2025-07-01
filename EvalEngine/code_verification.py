import os
import tvm
import numpy as np
import tvm.testing as testing
from tvm.runtime import load_module
from os.path import join,exists
import json
import logging
import time
from args import args
from tqdm import tqdm
import codebleu
from utils import search_op_info,get_code

logging.basicConfig(
    filename=f'/code/HPCTransEval/log/result_{args.model}_{args.op_type}.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def cal_codebleu(prediction,reference):
    result = codebleu.calc_codebleu([reference],[prediction],lang='c',weights=(0.25,0.25,0.25,0.25),tokenizer=None)
    return result
    
def compile_by_tvm(host_module, op, c_code):
    dev_module = host_module.imported_modules[0]
    dev_module.set_source(c_code)
    dev_module.compile_source()
    np.random.seed(42)
    dev = tvm.device(target.kind.name, 0)
    input_list = []
    output_list = []
    for _input_shape in op['input_shape']:
        input_list.append(np.random.random(tuple(_input_shape)).astype("float32"))
    for _output_shape in op['output_shape']:
        output_list.append(np.zeros(tuple(_output_shape)).astype("float32"))
    input_tvm_list = []
    output_tvm_list = []
    for _input in input_list:
        input_tvm_list.append(tvm.nd.array(_input,dev))
    for _output in output_list:
        output_tvm_list.append(tvm.nd.array(_output,dev))
    param_list = []
    param_list.extend(input_tvm_list)
    param_list.extend(output_tvm_list)
    return param_list, output_tvm_list

def exec_by_tvm(host_module, param_list):
    host_module(*param_list)

if __name__ == '__main__':
    base_path = 'YOUR_BENCHMARK_PATH'
    N_warmup = 100
    N_execute = 100

    target = tvm.target.Target(target="c", host="llvm")
    if not os.path.exists(base_path):
        raise Exception(f"{base_path} not found")
    
    with open(join(base_path,f'{args.op_type}.json'),'r') as file:
        op_info_dict = json.load(file)

    op_info = search_op_info(op_info_dict,args.file_name)
    # Correctness Test
    host_module = load_module(join(base_path,args.op_type,'host_module',args.file_name),'ll')
    device_module = load_module(join(base_path,args.op_type,'device_module',args.file_name),'c')
    host_module.import_module(device_module)

    # the c_code need to by verification
    c_code = op_info['c_code']
    c_code_generated = get_code(join(base_path,args.op_type,f'{args.model}_c',f'{args.file_name}.c'))

    # Separation of compilation and execution
    param_list, output_tvm_list = compile_by_tvm(host_module, op_info, c_code)
    exec_by_tvm(host_module, param_list)
    param_list_generated, output_tvm_list_generated = compile_by_tvm(host_module, op_info, c_code_generated)
    exec_by_tvm(host_module, param_list_generated)

    logging.info(f'{args.file_name} COMPILE PASS')
    try:
        assert len(output_tvm_list) == len(output_tvm_list_generated)
        for i in range(len(output_tvm_list)):
            testing.assert_allclose(output_tvm_list[i].numpy(), output_tvm_list_generated[i].numpy())
    except Exception as e:
        logging.info(f'{args.file_name} EXEC FAIL')
        exit(0)
    logging.info(f'{args.file_name} EXEC PASS')

    # Performance test
    param_list, output_tvm_list = compile_by_tvm(host_module, op_info, c_code)
    # warm up kernel
    for i in range(N_warmup):
        exec_by_tvm(host_module, param_list)
    c_code_start_time = time.time()
    for i in tqdm(range(N_execute)):
        exec_by_tvm(host_module, param_list)
    c_code_end_time = time.time()
    c_code_total_time = c_code_end_time - c_code_start_time
    param_list_generated, output_tvm_list_generated = compile_by_tvm(host_module, op_info, c_code_generated)
    # warm up kernel
    for i in range(N_warmup):
        exec_by_tvm(host_module, param_list_generated)
    gen_c_code_start_time = time.time()
    for i in tqdm(range(N_execute)):
        exec_by_tvm(host_module, param_list_generated)
    gen_c_code_end_time = time.time()
    gen_c_code_total_time = gen_c_code_end_time - gen_c_code_start_time
    with open(join('/code/HPCTransEval/results',f'{args.model}_{args.op_type}_c_time.jsonl'),'a+') as file:
        json.dump({args.file_name:c_code_total_time/N_execute},file)
    with open(join('/code/HPCTransEval/results',f'{args.model}_{args.op_type}_gen_c_time.jsonl'),'a+') as file:
        json.dump({args.file_name:gen_c_code_total_time/N_execute},file)