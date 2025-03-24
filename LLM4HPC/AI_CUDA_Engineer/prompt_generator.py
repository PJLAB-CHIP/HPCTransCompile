from os.path import join,exists
import os

class PromptGenerator:
    def __init__(self):
        self.template_base_path = '/code/LLM4HPC/AI_CUDA_Engineer/prompts'
        self.kernel_bench_base_path = '/code/LLM4HPC/AI_CUDA_Engineer/KernelBench'
        self.pytorch_code_functional_base_path = '/code/LLM4HPC/AI_CUDA_Engineer/PyTorch_Code_Functional'
        # self.cuda_code_base_path = '/code/LLM4HPC/AI_CUDA_Engineer/CUDA_Code'
        self.cuda_code_base_path = '/code/LLM4HPC/benchmark/cuda'
        # self.c_code_base_path = '/code/LLM4HPC/AI_CUDA_Engineer/CPU_Code'
        self.c_code_base_path = '/code/LLM4HPC/results/QwenCoder_14b'
        self.template_dict = {
            'conversion':{
                'BASE':'conversion_base_prompt_simplify.txt',
                'SYSTEM':'conversion_system_prompt.txt'
            },
            'optimization':{
                'BASE':'optimization_base_prompt.txt',
                'SYSTEM':'optimization_system_prompt.txt'
            },
            'translation':{
                'BASE':'translation_base_prompt.txt',
                'SYSTEM':'translation_system_prompt.txt'
            },
            'translation_c':{
                'BASE':'translation_c_base_prompt.txt',
                'SYSTEM':'translation_c_system_prompt.txt'
            }
        }
        self.save_path_dict = {
            'conversion': '/code/LLM4HPC/AI_CUDA_Engineer/PyTorch_Code_Functional',
            'translation': '/code/LLM4HPC/AI_CUDA_Engineer/CUDA_Code',
            'translation_c': '/code/LLM4HPC/results/QwenCoder_14b'
        }
        self.suffix_dict = {
            'conversion': '.py',
            'translation': '.cu',
            'translation_c': '.cpp'
        }

    def load_template(self,action='conversion'):
        try:
            with open(join(self.template_base_path,self.template_dict[action]['BASE'])) as file:
                base_template = file.read()
            with open(join(self.template_base_path,self.template_dict[action]['SYSTEM'])) as file:
                system_template = file.read()
        except FileNotFoundError:
            return 'File Not Found.'
        except Exception as e:
            return f'Error: {str(e)}'
        return base_template,system_template
    

    def load_source_code_single(self,level,operator,action='conversion',return_code=False):
        if action == 'conversion':
            file_path = join(self.kernel_bench_base_path,level,f'{operator}.py')
        elif action == 'translation':
            file_path = join(self.pytorch_code_functional_base_path,level,f'{operator}.py')
        elif action == 'translation_c':
            file_path = join(self.cuda_code_base_path,level,f'{operator}.cu')
        elif action == 'eval_c':
            file_path = join(self.c_code_base_path,level,f'{operator}.cpp')
        if return_code:
            with open(file_path,'r',encoding='utf-8') as file:
                code = file.read()
            return code
        else:
            return file_path
    
    def fill_template(self,source_code,action='conversion'):
        base_template,system_template = self.load_template(action)
        # TODO: 这部分仅做测试，逻辑后续完善
        if action == 'conversion':
            base_content = base_template.replace("{code}",source_code)
            system_content = system_template
        elif action == 'translation':
            base_content = base_template.replace("{EXAMPLES}",source_code)
            system_content = system_template
        elif action == 'translation_c':
            base_content = base_template.replace("{EXAMPLES}",source_code)
            system_content = system_template
        return base_content,system_content

    def load_source_code(self,levels=['level1']):
        """
        level: ['level1','level2','level3','level4']
        """
        source_code_dict = {}
        for level in levels:
            source_code_dict[level] = []
            level_folder_path = join(self.kernel_bench_base_path,level)
            operator_list = os.listdir(level_folder_path)
            for operator in operator_list:
                python_code = self.load_source_code_single(level,operator)
                source_code_dict[level].append(python_code)
        return source_code_dict


    def generate_prompt(self):
        pass

if __name__ == '__main__':
    prompt_generator = PromptGenerator()
    base_template,system_template = prompt_generator.load_template()
    python_code = prompt_generator.load_source_code_single('level1','1_Square_matrix_multiplication_.py')
    base_content,system_content = prompt_generator.fill_template(python_code)
    print(base_content)
    print('@@@@@@@@@@@@@@@@')
    print(system_content)