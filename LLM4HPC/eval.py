from turtle import forward
import torch
from torch.utils.cpp_extension import load_inline,load
from AI_CUDA_Engineer.prompt_generator import PromptGenerator

cpp_scr = ('torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);')

def evaluate(prompt_generator:PromptGenerator,level,operator,device,is_load_inline=False):
    if device == 'cuda':
        cuda_code_or_path = prompt_generator.load_source_code_single(level,operator,action='translation_c',return_code=is_load_inline)
        if is_load_inline:
            eval_module = load_inline(
                name=operator,
                cuda_sources=cuda_code_or_path,
                cpp_sources=cpp_scr,
                functions=["matmul_cuda"],
                verbose=True,
                )
        else:
            print(cuda_code_or_path)
            eval_module = load(
                name=operator,
                sources=[cuda_code_or_path],
                extra_cuda_cflags=["--expt-relaxed-constexpr"],
                verbose=True
            )
    elif device == 'cpu':
        cpu_code_or_path = prompt_generator.load_source_code_single(level,operator,action='eval_c',return_code=is_load_inline)
        if is_load_inline:
            eval_module = load_inline(
                name=operator,
                cpp_sources=cpu_code_or_path,
                functions=["matmul_cpu"],
                verbose=True
            )
        else:
            eval_module = load(
                name=operator,
                sources=cpu_code_or_path,
                verbose=True
            )
    return eval_module

def verify(prompt_generator:PromptGenerator,level,operator):
    # TODO:仅做测试样例，后续晚上自动化流程
    cuda_eval_module = evaluate(prompt_generator,level,operator,'cuda')
    cpu_eval_module = evaluate(prompt_generator,level,operator,'cpu')
    N = 32
    A = torch.rand(N, N, device='cuda', dtype=torch.float32)
    B = torch.rand(N, N, device='cuda', dtype=torch.float32)
    C_cuda = cuda_eval_module.matmul_cuda(A, B)
    A = A.to('cpu')
    B = B.to('cpu')
    C_cpu = cpu_eval_module.matmul_cpu(A, B)
    print(C_cuda)
    print(C_cpu)

if __name__ == '__main__':
    device = 'cpu'
    level = 'level1'
    # operator = '12_Matmul_with_diagonal_matrices_'
    operator = '22_Tanh'
    prompt_generator = PromptGenerator()

    # verify(prompt_generator,level,operator)

    eval_module = evaluate(prompt_generator,level,operator,device)
    N = 32
    A = torch.rand(N, device=device, dtype=torch.float32)
    B = torch.rand(N, N, device=device, dtype=torch.float32)
    C_cpu = eval_module.forward(A,B)
    print(C_cpu)