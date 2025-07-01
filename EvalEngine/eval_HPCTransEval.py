import os
from os.path import join,exists
import subprocess
from tqdm import tqdm
import argparse

BASE_PATH = 'YOUR_BENCHMARK_PATH'
parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str,default='qwencoder',
                    help='the model to evaluate')
parser.add_argument('--op_type',type=str,default='IndependentOperator',
                    help='the op type to generate')

args = parser.parse_args()

def extract_file_name_list(op_type):
    folder_path = join(BASE_PATH,op_type,'cuda')
    cu_files = [file[:-3] for file in os.listdir(folder_path) if file.endswith(".cu")]
    return cu_files

if __name__ == '__main__':
    file_name_list = extract_file_name_list(args.op_type)
    model = args.model
    op_type = args.op_type
    for file_name in tqdm(file_name_list):
        command = ["python", "PATH[code_verification.py]", f"--file_name={file_name}", f"--model={model}", f"--op_type={op_type}"]
        try:
            subprocess.run(command, check=True)
            print(f"Task {file_name} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Task {file_name} failed with error: {e}")