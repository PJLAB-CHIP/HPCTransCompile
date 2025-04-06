import os
from os.path import join,exists
import re

def extract_code(content,action):
    if action == 'translation_c' or action == 'optimization_c':
        match = re.search(r"<cpu>(.*?)</cpu>", content, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code.startswith("```cpp\n"):
                code = code[7:]  # 移除前缀
            if code.endswith("```"):
                code = code[:-3]  # 移除后缀
            return code
        else:
            match = re.search(r"```cpp\n(.*?)```", content, re.DOTALL)
            if match:
                code = match.group(1).strip()
                return code
            else:
                print('Fail to find <cpu></cpu> template.')
            # raise ValueError('Fail to find <cpu></cpu> template.')
    elif action == 'conversion':
        return content
    elif action == 'translation':
        match = re.search(r"<cuda>(.*?)</cuda>", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            print('Fail to find <cpu></cpu> template.')
            # raise ValueError('Fail to find <cpu></cpu> template.')

def save_to_file(content,file_name,suffix,action,save_path=None):
    code = extract_code(content,action)
    if save_path != None:
        file_path = join(save_path,f'{file_name}{suffix}')
        content_path = join(save_path,f'{file_name}.txt')
    else:
        file_path = f'{file_name}{suffix}'
        content_path = f'{file_name}.txt'

    # Content
    if content != None:
        with open(content_path,'w') as content_file:
            content_file.write(content)
    # Code
    code = extract_code(content,action)
    if code != None:
        with open(file_path,'w') as file:
            file.write(code)


def load_prompt(file_path):
    with open(file_path,'r',encoding='utf-8') as file:
        content = file.read()
    return content