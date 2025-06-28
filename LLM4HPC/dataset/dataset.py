from datasets import load_dataset
import pandas as pd
from os.path import join,exists 

base_path = '/code/LLM4HPC/benchmark/cuda'

def save_csv():
    ds = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive",cache_dir='/code/LLM4HPC/AI-CUDA-Engineer-Archive')
    ds['level_1'].to_pandas().to_csv('/code/LLM4HPC/benchmark/cuda/level1.csv',index=False)
    ds['level_2'].to_pandas().to_csv('/code/LLM4HPC/benchmark/cuda/level2.csv',index=False)
    ds['level_3'].to_pandas().to_csv('/code/LLM4HPC/benchmark/cuda/level3.csv',index=False)

def select_optimal_operator(level='level1'):
    df = pd.read_csv(join(base_path,f'{level}.csv'))
    filtered_df = df[df["Correct"] == True]
    result = filtered_df.loc[filtered_df.groupby("Op_Name")["CUDA_Runtime"].idxmin()]
    result.to_csv(join(base_path,f'{level}_optimal.csv'),index=False)
    for _operator in result.itertuples():
        _Op_Name = _operator.Op_Name
        _CUDA_Code = _operator.CUDA_Code
        with open(join(base_path,level,f'{_Op_Name}.cu'),'w') as file:
            file.write(_CUDA_Code)
    print(result)


if __name__ == '__main__':
    select_optimal_operator(level='level2')