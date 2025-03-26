```cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename scalar_t>
__global__ void expand_conv_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    int batch_size,
    int in_channels,
    int hidden_dim,
    int height,
    int width) {
    
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.z * blockDim.z + threadIdx.z;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c < hidden_dim && h < height && w < width) {
        scalar_t val = 0;
        for (int k = 0; k < in_channels; k++) {
            val += input[(blockIdx.w * in_channels + k) * height * width + h * width + w] * 
                   weight[c * in_channels + k];
        }
        output[(blockIdx.w * hidden_dim + c) * height * width + h * width + w] = val;
    }
}

template <typename scalar_t>
__global__ void depthwise_conv_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    int batch_size,
    int hidden_dim,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding) {
    
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int w_out = blockIdx.z * blockDim.z + threadIdx.z;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c < hidden_dim && h_out < (height + 2*padding - kernel_size)/stride + 1 && w_out < (width + 2*padding - kernel_size)/stride + 1) {
        scalar_t val = 0;
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    val += input[(blockIdx.w * hidden_dim + c) * height * width + h_in * width + w_in] * 
                           weight[c * kernel_size * kernel_size + kh * kernel_size + kw];
                }
            }
        }
        output[(blockIdx.w * hidden_dim + c) * ((height + 2*padding - kernel_size)/stride + 1) * 
               ((width + 2*padding - kernel_size)/stride + 1) + h_out * ((width + 2*padding - kernel_size)/stride + 1) + w_out] = val;
    }
}

template <typename scalar_t>
__global__ void project_conv_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    int batch_size,
    int hidden_dim,
    int out_channels,
    int height,
    int width) {
    
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.z * blockDim.z + threadIdx.z;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c < out_channels && h < height && w < width) {
        scalar_t val = 0;
        for (int k = 0; k < hidden_dim; k++) {
            val += input[(blockIdx.w * hidden_dim + k) * height * width + h * width + w] * 
                   weight[c * hidden_dim + k];
        }
        output[(blockIdx.w * out_channels + c) * height * width + h * width + w] = val;
    }
}

template <typename scalar_t>
__global__ void batchnorm_relu6_kernel(
    scalar_t* data,
    const scalar_t* running_mean,
    const scalar_t* running_var,
    const scalar_t* weight,
    const scalar_t* bias,
    int num_channels,
    int height,
    int width,
    bool training,
    bool inplace) {
    
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (c < num_channels && h < height && w < width) {
        scalar_t val = data[(blockIdx.w * num_channels + c) * height * width + h * width + w];
        scalar_t mean = running_mean[c];
        scalar_t var = running_var[c];
        scalar_t gamma = weight[c];
        scalar_t beta = bias[c];
        
        val = gamma * (val - mean) / sqrt(var + 1e-5) + beta;
        val = min(max(val, scalar_t(0)), scalar_t(6));
        
        data[(blockIdx.w * num_channels + c) * height * width + h * width + w] = val;
    }
}

template <typename scalar_t>
__global__ void residual_add_kernel(
    scalar_t* output,
    const scalar_t* identity,
    int num_channels,
    int height,
    int width) {
    
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (c < num_channels && h < height && w < width) {
        output[(blockIdx.w * num_channels + c) * height * width + h * width + w] += 
            identity[(blockIdx.w * num_channels + c) * height * width + h * width + w];
    }
}

torch::Tensor mbconv_forward(
    torch::Tensor input,
    torch::Tensor expand_conv_weight,
    torch::Tensor expand_conv_bn_weight,
    torch::Tensor expand_conv_bn_bias,
    torch::Tensor expand_conv_bn_running_mean,
    torch::Tensor expand_conv_bn_running_var,
    torch::Tensor depthwise_conv_weight,
    torch::Tensor depthwise_conv_bn_weight,
    torch::Tensor depthwise_conv_bn_bias,
    torch::Tensor depthwise_conv_bn_running_mean,
    torch::Tensor depthwise_conv_bn_running_var,
    torch::Tensor project_conv_weight,
    torch::Tensor project_conv_bn_weight,
    torch::Tensor project_conv_bn_bias,
    torch::Tensor project_conv_bn_running_mean,
    torch::Tensor project_conv_bn_running_var,
    bool use_residual,
    bool training) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(expand_conv_weight);
    CHECK_INPUT(expand_conv_bn_weight);
    CHECK_INPUT(expand_conv_bn_bias);
    CHECK_INPUT(expand_conv_bn_running_mean);
    CHECK_INPUT(expand_conv_bn_running_var);
    CHECK_INPUT(depthwise_conv_weight);
    CHECK_INPUT(depthwise_conv_bn_weight);
    CHECK_INPUT(depthwise_conv_bn_bias);
    CHECK_INPUT(depthwise_conv_bn_running_mean);
    CHECK_INPUT(depthwise_conv_bn_running_var);
    CHECK_INPUT(project_conv_weight);
    CHECK_INPUT(project_conv_bn_weight);
    CHECK_INPUT(project_conv_bn_bias);
    CHECK_INPUT(project_conv_bn_running_mean);
    CHECK_INPUT(project_conv_bn_running_var);
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int hidden_dim = depthwise_conv_weight.size(0);
    int out_channels = project_conv_weight.size(0);
    int kernel_size = depthwise_conv_weight.size(2);
    int stride = use_residual ? 1 : 2;
    
    auto identity = input.clone();
    auto output = input;
    
    // Expand convolution if needed
    if (expand_conv_weight.defined()) {
        auto expand_output = torch::zeros({batch_size, hidden_dim, height, width}, input.options());
        
        dim3 expand_block(32, 1, 1);
        dim3 expand_grid(
            (hidden_dim + expand_block.x - 1) / expand_block.x,
            (height + expand_block.y - 1) / expand_block.y,
            (width + expand_block.z - 1) / expand_block.z,
            batch_size);
        
        AT_DISPATCH_FLOATING_TYPES(input.type(), "expand_conv_kernel", ([&] {
            expand_conv_kernel<scalar_t><<<expand_grid, expand_block>>>(
                input.data<scalar_t>(),
                expand_conv_weight.data<scalar_t>(),
                expand_output.data<scalar_t>(),
                batch_size,
                in_channels,
                hidden_dim,
                height,
                width);
        }));
        
        output = expand_output;
        
        // BatchNorm + ReLU6
        dim3 bn_block(32, 8, 8);
        dim3 bn_grid(
            (hidden_dim + bn_block.x - 1) / bn_block.x,
            (height + bn_block.y - 1) / bn_block.y,
            (width + bn_block.z - 1) / bn_block.z,
            batch_size);
        
        AT_DISPATCH_FLOATING_TYPES(input.type(), "batchnorm_relu6_kernel", ([&] {
            batchnorm_relu6_kernel<scalar_t><<<bn_grid, bn_block>>>(
                output.data<scalar_t>(),
                expand_conv_bn_running_mean.data<scalar_t>(),
                expand_conv_bn_running_var.data<scalar_t>(),
                expand_conv_bn_weight.data<scalar_t>(),
                expand_conv_bn_bias.data<scalar_t>(),
                hidden_dim,
                height,
                width,
                training,
                true);
        }));
    }
    
    // Depthwise convolution
    int depthwise_height = (height + 2*(kernel_size/2) - kernel_size) / stride + 1;
    int depthwise_width = (width + 2*(kernel_size/2) - kernel_size) / stride + 1;
    auto depthwise_output = torch::zeros({batch_size, hidden_dim, depthwise_height, depthwise_width}, input.options());
    
    dim3 depthwise_block(32, 1, 1);
    dim3 depthwise_grid(
        (hidden_dim + depthwise_block.x - 1) / depthwise_block.x,
        (depthwise_height + depthwise_block.y - 1) / depthwise_block.y,
        (depthwise_width + depthwise_block.z - 1) / depthwise_block.z,
        batch_size);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv_kernel", ([&] {
        depthwise_conv_kernel<scalar_t><<<depthwise_grid, depthwise_block>>>(
            output.data<scalar_t>(),
            depthwise_conv_weight.data<scalar_t>(),
            depthwise_output.data<scalar_t>(),
            batch_size,
            hidden_dim,
            height,
            width,
            kernel_size,
            stride,
            kernel_size/2);
    }));
    
    output = depthwise_output;
    
    // Depthwise BatchNorm + ReLU6
    dim3 depthwise_bn_block(32, 8, 8);
    dim3 depthwise_bn_grid(
        (hidden_dim + depthwise_bn_block.x - 1) / depthwise_bn_block.x,
        (depthwise_height + depthwise_bn_block.y - 1) / depthwise_bn_block.y,
        (depthwise_width + depthwise_bn_block.z - 1) / depthwise_bn_block.z,
        batch_size);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "batchnorm_relu6_kernel", ([&] {
        batchnorm_relu6_kernel<scalar_t><<<depthwise_bn_grid, depthwise_bn_block>>>(
            output.data<scalar_t>(),
            depthwise_conv_bn_running_mean.data<scalar_t>(),
            depthwise_conv_bn_running_var.data<scalar_t>(),
            depthwise_conv_bn_weight.data<scalar_t>(),
            depthwise_conv_bn_bias.data<scalar_t>(),
            hidden_dim,
            depthwise_height,
            depthwise_width,
            training,
            true);
    }));
    
    // Project convolution
    auto project_output = torch::zeros({batch_size, out_channels, depthwise_height, depthwise_width}, input.options());
    
    dim3 project_block(32, 1, 1);
    dim3 project_grid(
        (out_channels + project_block.x - 1) / project_block.x,
        (depthwise_height + project_block.y - 1) / project_block.y,
        (depthwise_width + project_block.z - 1) / project_block.z,
        batch_size);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "project_conv_kernel", ([&] {
        project_conv_kernel<scalar_t><<<project_grid, project_block>>>(
            output.data<scalar_t>(),
            project_conv_weight.data<scalar_t>(),
            project_output.data<scalar_t>(),
            batch_size,
            hidden_dim,
            out_channels,
            depthwise_height,
            depthwise_width);
    }));
    
    output = project_output;
    
    // Project BatchNorm
    dim3 project_bn_block(32, 8, 8);
    dim3 project_bn_grid(
        (out_channels + project_bn_block.x - 1) / project_bn_block.x,
        (depthwise_height + project_bn_block.y - 1) / project_bn_block.y,
        (depthwise_width + project_bn_block.z - 1) / project_bn_block.z,
        batch_size);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "batchnorm_relu6_kernel", ([&] {
        batchnorm_relu6_kernel<scalar_t><<<project_bn_grid, project_bn_block>>>(
            output.data<scalar_t>(),
            project_conv_bn_running_mean.data<scalar_t>(),
            project_conv_bn_running_var.data<scalar_t>(),
            project_conv_bn_weight.data<scalar_t>(),
            project_conv_bn_bias.data<scalar_t>(),
            out_channels,
            depthwise_height,
            depthwise_width,
            training,
            false);
    }));
    
    // Residual connection
    if (use_residual) {
        dim3 residual_block(32, 8, 8);
        dim3 residual_grid(
            (out_channels + residual_block.x - 1) / residual_block.x,
            (depthwise_height + residual_block.y - 1) / residual_block.y,
            (depthwise_width + residual_block.z - 1) / residual_block.z,
            batch_size);
        
        AT_DISPATCH_FLOATING_TYPES(input.type(), "residual_add_kernel", ([&] {
            residual_add_kernel<scalar_t><<<residual_grid, residual_block>>>(
                output.data<scalar_t>(),
                identity.data<scalar_t>(),
                out_channels,
                depthwise_height,
                depthwise_width);
        }));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mbconv_forward, "MBConv forward (CUDA)");
}
```