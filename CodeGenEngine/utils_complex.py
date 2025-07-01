import os
from tvm import (te,
                 auto_scheduler,
                 runtime,
                 topi)
import json
import tvm
import random
from os.path import join,exists
import pandas as pd


class CodeGen:
    def __init__(self, search_time, log_file_path,verbose_key=False) -> None:
        self.search_time = search_time
        self.log_file_path = log_file_path
        self.verbose_key = verbose_key

    def c_codegen(self, topi_ops=None, op_args=None):
        if topi_ops is None:
            raise ValueError("topi_ops must not be None and needs to be provided when c_codegen.")
        if op_args is None:
            raise ValueError("op_args must not be None and needs to be provided when c_codegen.")

        # step1: assign search task
        target_llvm = tvm.target.Target(target="llvm", host="llvm")
        task_c = tvm.auto_scheduler.SearchTask(func=topi_ops, args=(*op_args, ), target=target_llvm)

        os.makedirs(join(self.log_file_path,'c'),exist_ok=True)
        c_log_file_path = join(self.log_file_path,'c',f'{topi_ops.__name__}_c.log')
        if exists(c_log_file_path):
            with open(c_log_file_path,'w') as file:
                pass

        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=self.search_time,
            measure_callbacks=[auto_scheduler.RecordToFile(c_log_file_path)],
            verbose=self.verbose_key,
        )

        # Run auto-tuning (search)
        task_c.tune(tune_option)

        # Apply the best schedule
        target_c = tvm.target.Target(target="c", host="c")
        sch, args = task_c.apply_best(c_log_file_path)
        c_module = tvm.build(sch, args, target_c)
        c_code = c_module.get_source()
        print(c_code)
        lowered_ir = tvm.lower(sch, args, simple_mode=True)

        return c_code, str(lowered_ir)
    
    def cuda_codegen(self, topi_ops=None, op_args=None):
        if topi_ops is None:
            raise ValueError("topi_ops must not be None and needs to be provided when cuda_codegen.")
        if op_args is None:
            raise ValueError("op_args must not be None and needs to be provided when cuda_codegen.")

        target_cuda = tvm.target.Target("cuda")
        task_cuda = tvm.auto_scheduler.SearchTask(func=topi_ops, args=(*op_args, ), target=target_cuda)

        cuda_log_file_path = join(self.log_file_path,'cuda',f'{topi_ops.__name__}_cuda.log')
        if exists(cuda_log_file_path):
            with open(cuda_log_file_path,'w') as file:
                pass

        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=self.search_time,
            measure_callbacks=[auto_scheduler.RecordToFile(cuda_log_file_path)],
            verbose=self.verbose_key,
        )

        # Run auto-tuning (search)
        task_cuda.tune(tune_option)

        # Apply the best schedule
        sch, args = task_cuda.apply_best(cuda_log_file_path)

        cuda_module = tvm.build(sch, args, "cuda")
        cuda_code = cuda_module.imported_modules[0].get_source()
        return cuda_code
    
class ArgsGen:
    def __init__(self,param_list) -> None:
        self.param_list = param_list
        # record random values for each param to avoid duplication
        self.param_record_list = []
        self.params = {}

    def check_redundancy(self,temp_param_record_list):
        if temp_param_record_list in self.param_record_list:
            return True
        else:
            return False
        
    def select_param(self,param_name):
        """
            N: batch size
            H: height
            W: width
            CO: output channels
            CI: input channels
            KH: kernel height
            KW: kernel width
            stride: stride
            padding: padding
        """
        if param_name == 'CO' or param_name == 'CI':
            param_val = random.randint(10, 100)
        elif param_name == 'KH' or param_name == 'KW':
            param_val = random.randint(1, 20)
        elif param_name == 'strides':
            param_val = (random.randint(1, 3), random.randint(1, 3))
        elif param_name == 'padding':
            param_val = (random.randint(1, 1), random.randint(1, 1))
        else:
            param_val = random.randint(1, 20)
        return param_val
    
    def randomize_params(self):
        # randomly generate params
        temp_param_record_list = []
        for param_name in self.param_list:
            param_val = self.select_param(param_name)
            temp_param_record_list.append(param_val)
            self.params[param_name] = param_val
        while self.check_redundancy(temp_param_record_list):
            temp_param_record_list = []
            for param_name in self.param_list:
                param_val = self.select_param(param_name)
                temp_param_record_list.append(param_val)
                self.params[param_name] = param_val
        self.param_record_list.append(temp_param_record_list)
    
    def get_param_values(self):
        return [value for value in self.params.values()]

def write_ops_to_json(op_name, c_code, cuda_code, ir_code, op_args, save_file_path = './generate/generation.json'):
    op_data = {
        'op_name': op_name,
        'c_code': c_code,
        'cuda_code': cuda_code,
        'ir_code': ir_code,
        'op_args':op_args
    }
    with open(save_file_path, 'a') as file:
        json.dump(op_data,file)

def write_ops_to_file(op_name, c_code, cuda_code, ir_code, op_args, save_file_path='./generate'):

    with open(join(save_file_path,'c',f'{op_name}_{op_args}.c'),'w') as file:
        file.write(c_code)

    with open(join(save_file_path,'ir',f'{op_name}_{op_args}.ir'),'w') as file:
        file.write(ir_code)

    with open(join(save_file_path,'cuda',f'{op_name}_{op_args}.cu'),'w') as file:
        file.write(cuda_code)

def handle_timeout(signum, frame):
    raise TimeoutError("Execution timed out")

def load_ops_from_txt(file_path):
    ops_dict = {
        'op':[],
        'description':[]
    }
    with open(file_path,'r') as f:
        for index,line in enumerate(f):
            if index % 2 == 0 and index % 4 != 0:
                ops_dict['description'].append(line.strip())
            if index % 4 == 0:
                ops_dict['op'].append(line.strip())
    ops_df = pd.DataFrame(ops_dict)
    ops_df.to_csv('/code/OpGen-Verify/code_generation/ops/topi_ops.csv',index=False)

def parse_func(func_str):
    necessary_params = []
    optional_params = []
    assert len(func_str.split('(')) == 2
    func_name = func_str.split('(')[0]
    params_list = func_str.split('(')[-1][:-1]

    if '[' in params_list and ']' in params_list:
        optional_params_str = params_list.split('[')[-1][:-1]
        optional_params = optional_params_str.split(',')
        optional_params = [s.strip() for s in optional_params if s.strip() and '.' not in s]
    else:
        optional_params = []

    if '[' in params_list and ']' in params_list:
        necessary_params_str = params_list.split('[')[0]
    else:
        necessary_params_str = params_list
    necessary_params = necessary_params_str.split(',')
    necessary_params = [s.strip() for s in necessary_params if s.strip() and '.' not in s]
    print(func_name,':',params_list,'|',necessary_params,'|',optional_params)
    return func_name,necessary_params,optional_params


if __name__ == '__main__':
    ops_df = pd.read_csv('/code/OpGen-Verify/code_generation/ops/topi_ops.csv')
    op_list = []
    func_list = ops_df['op']
    func_description = ops_df['description']
    for func,description in zip(func_list,func_description):
        func_name,necessary_params,optional_params = parse_func(func)
        op_list.append({
            'op_name':func_name,
            'necessary_params':necessary_params,
            'optional_params':optional_params,
            'description':description
        })
    with open('/code/OpGen-Verify/code_generation/ops/topi_ops.json','w') as f:
        json.dump(op_list,f,indent=4)