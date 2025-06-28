import torch
from torch.utils.cpp_extension import load_inline
from AI_CUDA_Engineer.prompt_generator import PromptGenerator

# cpp_scr = 'torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);'

# if __name__ == '__main__':
#     level = 'level1'
#     operator = '1_Square_matrix_multiplication_'
#     prompt_generator = PromptGenerator()
#     cuda_code = prompt_generator.load_source_code_single(level,operator,action='eval')
#     print(cuda_code)

#     # 使用 load_inline 编译和加载模块
#     matmul_module = load_inline(
#         name="matmul_tiled",
#         cuda_sources=cuda_code,
#         cpp_sources=cpp_scr,
#         functions=["matmul_cuda"],
#         verbose=True,
#     )

#     # 测试模块
#     N = 32
#     A = torch.rand(N, N, dtype=torch.float32)
#     B = torch.rand(N, N, dtype=torch.float32)
#     C = matmul_module.forward(A, B)

#     print("Result of C:")
#     print(C)

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
        out[idx] += 1;
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    cudaDeviceSynchronize();

    return out;
}
"""

cpp_src = (
    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
)

cpp_code = """
#include <vector>

// C++ function for element-wise addition
void add_cpu(const float* a, const float* b, float* c, int size) {
    std::cout << "Size: " << size << std::endl;
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}
"""

cpp_code = """
#include <torch/extension.h>
#include <iostream>

torch::Tensor elementwise_add_cpu(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(a.device().is_cpu() && b.device().is_cpu(), "Input tensors must be on CPU");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "Input tensors must be float32");

    auto size = a.numel();
    auto out = torch::empty_like(a);

    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    std::cout << "Running addition with size: " << size << std::endl;

    for (int i = 0; i < size; ++i) {
        out_ptr[i] = a_ptr[i] + b_ptr[i];
        out_ptr[i] += 1;
    }

    return out;
}

"""


# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

elementwise_add_cpu = load_inline(
    name="elementwise_add_cpu",
    cpp_sources=cpp_code,
    functions=["elementwise_add_cpu"],
    verbose=True
)


"""TEST CUDA"""
# N = 1024
# A = torch.ones(N,N,device='cuda',dtype=torch.float32)
# B = torch.ones(N,N,device='cuda',dtype=torch.float32)
# print(A)
# print(B)
# C = elementwise_add.elementwise_add_cuda(A,B)
# print("Result of C:")
# print(C)

"""TEST CPU"""

N = 1024
A = torch.ones(N,N,device='cpu',dtype=torch.float32)
B = torch.ones(N,N,device='cpu',dtype=torch.float32)
print(A)
print(B)
C = elementwise_add_cpu.elementwise_add_cpu(A,B)
print("Result of C:")
print(C)

# size = 1024
# a = torch.ones(size, device="cpu", dtype=torch.float32)
# b = torch.ones(size, device="cpu", dtype=torch.float32)
# c = torch.empty(size, device="cpu", dtype=torch.float32)

# # 调用 C++ 函数
# elementwise_add_cpu.add_cpu(a.data_ptr(), b.data_ptr(), c.data_ptr(), size)
# print(torch.allclose(a + b, c))
