```cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declaration of the CUDA kernel
void conv3d_cuda_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    int stride,
    int padding,
    int dilation,
    int groups);

// PyTorch interface
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    // Check input dimensions
    TORCH_CHECK(input.dim() == 5, "Input must be 5D (batch, channels, height, width, depth)");
    TORCH_CHECK(weight.dim() == 5, "Weight must be 5D (out_channels, in_channels/groups, kH, kW, kD)");
    
    // Calculate output dimensions
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t input_height = input.size(2);
    int64_t input_width = input.size(3);
    int64_t input_depth = input.size(4);
    
    int64_t out_channels = weight.size(0);
    int64_t kernel_height = weight.size(2);
    int64_t kernel_width = weight.size(3);
    int64_t kernel_depth = weight.size(4);
    
    int64_t output_height = (input_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    int64_t output_width = (input_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;
    int64_t output_depth = (input_depth + 2 * padding - dilation * (kernel_depth - 1) - 1) / stride + 1;
    
    // Create output tensor
    auto output = torch::zeros(
        {batch_size, out_channels, output_height, output_width, output_depth},
        input.options());
    
    // Call CUDA kernel
    conv3d_cuda_forward(
        input,
        weight,
        bias ? *bias : torch::Tensor(),
        output,
        stride,
        padding,
        dilation,
        groups);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D convolution forward (CUDA)");
}
```

The actual CUDA kernel implementation would need to be written in a separate .cu file, but here's the basic structure it would follow:

```cpp
// conv3d_kernel.cu
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int input_depth,
    int output_height,
    int output_width,
    int output_depth,
    int kernel_height,
    int kernel_width,
    int kernel_depth,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    // Implementation would go here
    // This would involve:
    // 1. Calculating the current output position
    // 2. Iterating over the kernel dimensions
    // 3. Applying the convolution operation
    // 4. Adding bias if present
}

void conv3d_cuda_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    // Get tensor dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int input_depth = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);
    int kernel_depth = weight.size(4);
    
    // Calculate grid and block dimensions
    dim3 blocks(32, 32);  // Adjust based on your needs
    dim3 threads(
        (output.size(2) + blocks.x - 1) / blocks.x,
        (output.size(3) + blocks.y - 1) / blocks.y,
        (output.size(4) + 1 - 1) / 1);
    
    // Launch kernel
    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        input_depth,
        output.size(2),
        output.size(3),
        output.size(4),
        kernel_height,
        kernel_width,
        kernel_depth,
        stride,
        padding,
        dilation,
        groups);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in conv3d_kernel: %s\n", cudaGetErrorString(err));
    }
}
```