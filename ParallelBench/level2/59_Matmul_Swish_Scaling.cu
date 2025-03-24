#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void swish_scaling_kernel_2d(const float* __restrict__ input, float* output, float scaling_factor, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        float x = input[idx];
        // Swish activation: x * sigmoid(x)
        float sigmoid = 1.0f / (1.0f + expf(-x));
        float y = x * sigmoid * scaling_factor;
        output[idx] = y;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double scaling_factor) {

    // Ensure tensors are contiguous
    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    // Ensure tensors are on CUDA
    TORCH_CHECK(x.is_cuda(), "Input tensor 'x' must be a CUDA tensor.");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor.");
    TORCH_CHECK(bias.is_cuda(), "Bias tensor must be a CUDA tensor.");

    // Ensure data types are float32
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Input tensor 'x' must be of type torch.float32.");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "Weight tensor must be of type torch.float32.");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "Bias tensor must be of type torch.float32.");

    // Compute linear transformation: y = x @ weight.T + bias
    auto y = at::addmm(bias, x, weight.t());

    // Get the dimensions
    int rows = y.size(0);
    int cols = y.size(1);

    // Allocate output tensor
    auto output = at::empty_like(y);

    // Launch the CUDA kernel
    dim3 threads(32, 32);
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

    swish_scaling_kernel_2d<<<blocks, threads>>>(
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(scaling_factor),
        rows,
        cols);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Custom CUDA forward function");
}
