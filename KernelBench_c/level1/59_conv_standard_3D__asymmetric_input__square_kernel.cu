#include <cstdio>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

    int index = blockIdx.z * blockDim.x * blockDim.y * blockDim.z +
            threadIdx.z * blockDim.x * blockDim.y +
            threadIdx.y * blockDim.x +
            threadIdx.x;

    const int total_elements = depth_out * height_out * width_out;
    if (index >= total_elements) return;

    const int ow = index % width_out;
    const int oh = (index / width_out) % height_out;
    const int od = index / (width_out * height_out);

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
                        const int weight_idx = (oc * (in_channels / groups) + ic) * kernel_size * kernel_size * kernel_size + kd * kernel_size * kernel_size + kh * kernel_size + kw;
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

torch::Tensor conv3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
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

    int out_d = (depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto options = input.options();
    auto output = torch::zeros(
        {batch_size, out_channels, out_d, out_h, out_w},
        input.options());

    // Get pointers to tensor data
    const float* input_data = input.data_ptr<float>();
    const float* weight_data = weight.data_ptr<float>();
    const float* bias_data = bias.defined() ? bias.data_ptr<float>() : nullptr;
    float* output_data = output.data_ptr<float>();

    // Set grid and block dimensions
    int block_dim = 8;
    dim3 block(block_dim, block_dim, block_dim);
    dim3 grid(
        batch_size,
        out_channels,
        (out_d * out_h * out_w + block.x * block.y * block.z - 1) / (block.x * block.y * block.z)
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
        out_d,
        out_h,
        out_w
    );

    // Synchronize to check for errors
    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv3d_forward_cuda, "3D convolution forward (CUDA)");
}