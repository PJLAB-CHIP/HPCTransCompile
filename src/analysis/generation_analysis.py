import os
from os.path import join,exists
import math

def calculate_mean(lst):
    if len(lst) == 0:
        return 0  # 处理空列表的情况，返回默认值或者引发异常

    total = sum(lst)
    mean = total / len(lst)
    return mean

if __name__ == '__main__':
    FILE_NAME = '/code/LLM4HPCTransCompile/evaluate/CodeLlama-13b-hf-400-v3.0_topi_without_ir_by_name_simplify-v3.0_topi_without_ir_by_name_simplify_3_15/codebleu.txt'
    
    op_score_dict = {'acosh':[],'greater':[],'cos':[]}
    with open(FILE_NAME,'r') as file:
        for line in file:
            split_str = line.split(':',1)
            if len(split_str) == 2:
                op_name = (split_str[0].strip()).split('_')[0]
                if op_name in ['acosh','greater','cos']:
                    score = eval(split_str[1].strip())
                    op_score_dict[op_name].append(score['codebleu'])
    print(calculate_mean(op_score_dict['acosh']))
    print(calculate_mean(op_score_dict['cos']))
    print(calculate_mean(op_score_dict['greater']))