import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from os.path import join,exists
import json
import argparse
import pandas as pd
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,default='QwenCoder_14b_75')
    parser.add_argument('--level',type=str,default='level3')
    return parser.parse_args()

args = parse_args()

all_operators_path = join('/code/KernelBench/KernelBench',args.level)
all_operators = [os.path.splitext(file)[0] for file in os.listdir(all_operators_path) if os.path.isfile(join(all_operators_path,file))]

def select_operator(key):
    for operator in all_operators:
        if operator.split('_')[0] == key:
            return operator
    raise ValueError(f'Not found {key}.')

if __name__ == '__main__':
    result_dict = {
        'operator':[],
        'compiled':[],
        'correctness':[],
        'runtime':[],
        'torch_runtime':[],
        'accelerate_rate':[]
    }
    with open(join('/code/LLM4HPCTransCompile/Results',args.model_name,f'{args.level[-1]}_eval_results.json'),'r',encoding='utf-8') as file:
        data = json.load(file)
    for key,val in data.items():
        result_dict['operator'].append(select_operator(key))
        if val['compiled']:
            result_dict['compiled'].append(True)
            if val['correctness']:
                result_dict['correctness'].append(True)
                result_dict['runtime'].append(val['runtime'])
                result_dict['torch_runtime'].append(val['torch_runtime'])
                result_dict['accelerate_rate'].append(round(val['torch_runtime'] / val['runtime'], 3))
            else:
                result_dict['correctness'].append(False)
                result_dict['runtime'].append(-1)
                result_dict['torch_runtime'].append(-1)
                result_dict['accelerate_rate'].append(-1)
        else:
            result_dict['compiled'].append(False)
            result_dict['correctness'].append(False)
            result_dict['runtime'].append(-1)
            result_dict['torch_runtime'].append(-1)
            result_dict['accelerate_rate'].append(-1)
    df = pd.DataFrame(result_dict)
    df.to_csv(join('/code/LLM4HPCTransCompile/Results',args.model_name,f'{args.level[-1]}_eval_results.csv'),index=False)

            
