#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void sigmoid_scaling_residual_add_kernel(scalar_t* x_data, const scalar_t* original_x_data, scalar_t scaling_factor, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements per thread when possible
    #pragma unroll
    for (int i = tid * 4; i < size - 3; i += stride * 4) {
        scalar_t val1 = x_data[i];
        scalar_t val2 = x_data[i + 1];
        scalar_t val3 = x_data[i + 2];
        scalar_t val4 = x_data[i + 3];
        
        scalar_t orig1 = original_x_data[i];
        scalar_t orig2 = original_x_data[i + 1];
        scalar_t orig3 = original_x_data[i + 2];
        scalar_t orig4 = original_x_data[i + 3];
        
        // Compute sigmoid and scale
        val1 = scalar_t(1.0) / (scalar_t(1.0) + exp(-val1));
        val2 = scalar_t(1.0) / (scalar_t(1.0) + exp(-val2));
        val3 = scalar_t(1.0) / (scalar_t(1.0) + exp(-val3));
        val4 = scalar_t(1.0) / (scalar_t(1.0) + exp(-val4));
        
        val1 = val1 * scaling_factor + orig1;
        val2 = val2 * scaling_factor + orig2;
        val3 = val3 * scaling_factor + orig3;
        val4 = val4 * scaling_factor + orig4;
        
        x_data[i] = val1;
        x_data[i + 1] = val2;
        x_data[i + 2] = val3;
        x_data[i + 3] = val4;
    }
    
    // Handle remaining elements
    for (int i = tid * 4 + (size / 4) * 4; i < size; i += stride) {
        scalar_t val = x_data[i];
        scalar_t orig_val = original_x_data[i];
        val = scalar_t(1.0) / (scalar_t(1.0) + exp(-val));
        val = val * scaling_factor + orig_val;
        x_data[i] = val;
    }
}

void sigmoid_scaling_residual_add_cuda(torch::Tensor& x, const torch::Tensor& original_x, float scaling_factor)
{
    const int threads = 256;
    const int size = x.numel();
    const int blocks = min(65535, (size + threads * 4 - 1) / (threads * 4));

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "sigmoid_scaling_residual_add_cuda", ([&] {
        sigmoid_scaling_residual_add_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            original_x.data_ptr<scalar_t>(),
            static_cast<scalar_t>(scaling_factor),
            size);
    }));
}

torch::Tensor module_fn_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaling_factor)
{
    TORCH_CHECK(x.is_cuda(), "Input tensor x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias tensor must be a CUDA tensor");

    x = torch::addmm(bias, x, weight.t());
    torch::Tensor original_x = x.clone();
    sigmoid_scaling_residual_add_cuda(x, original_x, scaling_factor);
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &module_fn_forward, "Module function forward (CUDA)");
}