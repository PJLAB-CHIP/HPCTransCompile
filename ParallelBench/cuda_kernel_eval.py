from turtle import forward
import torch
from torch.utils.cpp_extension import load_inline,load

cpp_scr = ('torch::Tensor forward(torch::Tensor A, torch::Tensor B);')

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

torch::Tensor forward(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    cudaDeviceSynchronize();

    return out;
}
"""

if __name__ == '__main__':
    eval_module = load_inline(
        name='test',
        cuda_sources=source,
        cpp_sources=cpp_scr,
        functions=["forward"],
        verbose=True,
        )
    
    N = 1024
    A = torch.ones(N,device='cuda',dtype=torch.float32)
    B = torch.ones(N,device='cuda',dtype=torch.float32)
    C = eval_module.forward(A,B)
    print(C)
