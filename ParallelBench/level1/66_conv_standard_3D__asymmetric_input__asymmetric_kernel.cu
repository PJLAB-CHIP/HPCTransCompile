```cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declaration of the CUDA kernel
void conv3d_forward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups);

// C++ wrapper function
torch::Tensor conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups) {
    
    // Check input dimensions
    TORCH_CHECK(input.dim() == 5, "Input must be 5D (batch, channels, depth, height, width)");
    TORCH_CHECK(weight.dim() == 5, "Weight must be 5D (out_channels, in_channels/groups, kd, kh, kw)");
    
    // Calculate output dimensions
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t depth = input.size(2);
    int64_t height = input.size(3);
    int64_t width = input.size(4);
    
    int64_t out_channels = weight.size(0);
    int64_t kd = weight.size(2);
    int64_t kh = weight.size(3);
    int64_t kw = weight.size(4);
    
    int64_t pad_d = padding[0];
    int64_t pad_h = padding[1];
    int64_t pad_w = padding[2];
    
    int64_t stride_d = stride[0];
    int64_t stride_h = stride[1];
    int64_t stride_w = stride[2];
    
    int64_t dilation_d = dilation[0];
    int64_t dilation_h = dilation[1];
    int64_t dilation_w = dilation[2];
    
    int64_t depth_out = (depth + 2 * pad_d - dilation_d * (kd - 1) - 1) / stride_d + 1;
    int64_t height_out = (height + 2 * pad_h - dilation_h * (kh - 1) - 1) / stride_h + 1;
    int64_t width_out = (width + 2 * pad_w - dilation_w * (kw - 1) - 1) / stride_w + 1;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, 
                             input.options());
    
    // Call CUDA implementation
    conv3d_forward_cuda(input, weight, bias, output, stride, padding, dilation, groups);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv3d_forward, "3D Convolution forward (CUDA)");
}
```

The actual CUDA kernel implementation would be in a separate .cu file with the following structure:

```cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int depth,
    const int height,
    const int width,
    const int depth_out,
    const int height_out,
    const int width_out,
    const int kd,
    const int kh,
    const int kw,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int dilation_d,
    const int dilation_h,
    const int dilation_w,
    const int groups) {
    
    // 3D grid of threads
    int n = blockIdx.x;
    int oc = blockIdx.y;
    int od = blockIdx.z * blockDim.z + threadIdx.z;
    int oh = blockIdx.z * blockDim.y + threadIdx.y;
    int ow = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (n >= batch_size || oc >= out_channels || od >= depth_out || oh >= height_out || ow >= width_out) {
        return;
    }
    
    float value = 0.0f;
    
    // Compute input position
    int id_start = od * stride_d - pad_d;
    int ih_start = oh * stride_h - pad_h;
    int iw_start = ow * stride_w - pad_w;
    
    // Iterate over kernel
    for (int kd_idx = 0; kd_idx < kd; ++kd_idx) {
        int id = id_start + kd_idx * dilation_d;
        if (id < 0 || id >= depth) continue;
        
        for (int kh_idx = 0; kh_idx < kh; ++kh_idx) {
            int ih = ih_start + kh_idx * dilation_h;
            if (ih < 0 || ih >= height) continue;
            
            for (int kw_idx = 0; kw_idx < kw; ++kw_idx) {
                int iw = iw_start + kw_idx * dilation_w;
                if (iw < 0 || iw >= width) continue;
                
                // Iterate over input channels in group
                for (int ic = 0; ic < in_channels / groups; ++ic) {
                    int input_idx = ((n * in_channels + (oc / (out_channels / groups)) * (in_channels / groups) + ic) * depth + id) * height * width + ih * width + iw;
                    int weight_idx = ((oc * (in_channels / groups) + ic) * kd + kd_idx) * kh * kw + kh_idx * kw + kw_idx;
                    
                    value += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias if present
    if (bias != nullptr) {
        value += bias[oc];
    }
    
    // Write output
    int output_idx = ((n * out_channels + oc) * depth_out + od) * height_out * width_out + oh * width_out + ow;
    output[output_idx] = value;
}

void conv3d_forward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups) {
    
    // Get tensor dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    
    int out_channels = weight.size(0);
    int kd = weight.size(2);
    int kh = weight.size(3);
    int kw = weight.size(4);
    
    int depth_out = output.size(2);
    int height_out = output.size(3);
    int width_out = output.size(4);
    
    // Get pointers to data
    const float* input_data = input.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    const float* bias_data = bias.defined() ? bias.data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();
    
    // Configure CUDA kernel launch
    dim3 block(8, 8, 8);  // Threads per block
    dim3 grid(
        (batch_size + block.x - 1) / block.x,
        (out_channels + block.y - 1) / block.y,
        (depth_out * height_out * width_out + block.z - 1) / block.z
    );
    
    // Launch kernel
    conv3d_kernel<<<grid, block>>>(
        input_data, weight_data, bias_data, output_data,
        batch_size, in_channels, out_channels,
        depth, height, width,
        depth_out, height_out, width_out,
        kd, kh, kw,
        padding[0], padding[1], padding[2],
        stride[0], stride[1], stride[2],
        dilation[0], dilation[1], dilation[2],
        groups
    );
    
    // Synchronize to check for errors
    cudaDeviceSynchronize();
    AT_ASSERT(cudaGetLastError() == cudaSuccess);
}
```