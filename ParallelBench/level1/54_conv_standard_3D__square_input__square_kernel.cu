```cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

// CUDA kernel declarations
void conv3d_forward_cuda(
    const at::Tensor input,
    const at::Tensor weight,
    const at::Tensor bias,
    at::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups
);

// Function to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Forward function
at::Tensor conv3d_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) {
        CHECK_INPUT(bias);
    }

    // Calculate output dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int depth_out = (depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int height_out = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int width_out = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Create output tensor
    auto output = at::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());

    // Call CUDA kernel
    conv3d_forward_cuda(
        input,
        weight,
        bias,
        output,
        stride,
        padding,
        dilation,
        groups
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv3d_forward, "3D Convolution forward (CUDA)");
}
```

```cu
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for 3D convolution
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
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int depth_out,
    const int height_out,
    const int width_out
) {
    // Calculate output position
    const int n = blockIdx.x;
    const int oc = blockIdx.y;
    const int od = blockIdx.z * blockDim.z + threadIdx.z;
    const int oh = blockIdx.z * blockDim.y + threadIdx.y;
    const int ow = blockIdx.z * blockDim.x + threadIdx.x;

    if (n >= batch_size || oc >= out_channels || od >= depth_out || oh >= height_out || ow >= width_out) {
        return;
    }

    // Calculate input position
    const int id_start = od * stride - padding;
    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;

    float value = 0.0f;

    // Perform convolution
    for (int ic = 0; ic < in_channels / groups; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    const int id = id_start + kd * dilation;
                    const int ih = ih_start + kh * dilation;
                    const int iw = iw_start + kw * dilation;

                    if (id >= 0 && id < depth && ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        const int input_idx = ((n * in_channels + (oc / (out_channels / groups)) * (in_channels / groups) + ic) * depth + id) * height * width + ih * width + iw;
                        const int weight_idx = ((oc * (in_channels / groups) + ic) * kernel_size * kernel_size * kernel_size + kd * kernel_size * kernel_size + kh * kernel_size + kw;
                        value += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Add bias if present
    if (bias != nullptr) {
        value += bias[oc];
    }

    // Write output
    const int output_idx = ((n * out_channels + oc) * depth_out + od) * height_out * width_out + oh * width_out + ow;
    output[output_idx] = value;
}

void conv3d_forward_cuda(
    const at::Tensor input,
    const at::Tensor weight,
    const at::Tensor bias,
    at::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    // Get tensor dimensions
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int depth = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);
    
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int depth_out = output.size(2);
    const int height_out = output.size(3);
    const int width_out = output.size(4);

    // Get pointers to tensor data
    const float* input_data = input.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    const float* bias_data = bias.defined() ? bias.data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();

    // Set grid and block dimensions
    dim3 block(16, 16, 1);
    dim3 grid(
        batch_size,
        out_channels,
        (depth_out * height_out * width_out + block.x * block.y * block.z - 1) / (block.x * block.y * block.z)
    );

    // Launch kernel
    conv3d_kernel<<<grid, block>>>(
        input_data,
        weight_data,
        bias_data,
        output_data,
        batch_size,
        in_channels,
        out_channels,
        depth,
        height,
        width,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        depth_out,
        height_out,
        width_out
    );

    // Synchronize to check for errors
    cudaDeviceSynchronize();
}
```