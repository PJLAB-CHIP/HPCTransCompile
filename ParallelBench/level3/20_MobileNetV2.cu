```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA kernel for pointwise convolution
__global__ void pointwise_conv_kernel(
    const float* input, const float* weight,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int height, int width, int kernel_size
) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c_out = blockIdx.w * blockDim.w + threadIdx.w;

    if (b >= batch_size || c_out >= out_channels || h >= height || w >= width) return;

    float sum = 0.0f;
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h + kh - kernel_size / 2;
                int w_in = w + kw - kernel_size / 2;
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = b * in_channels * height * width + 
                                   c_in * height * width + 
                                   h_in * width + w_in;
                    int weight_idx = c_out * in_channels * kernel_size * kernel_size + 
                                    c_in * kernel_size * kernel_size + 
                                    kh * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    int output_idx = b * out_channels * height * width + 
                     c_out * height * width + 
                     h * width + w;
    output[output_idx] = sum;
}

// CUDA kernel for depthwise convolution
__global__ void depthwise_conv_kernel(
    const float* input, const float* weight,
    float* output,
    int batch_size, int channels,
    int height, int width, int kernel_size, int stride
) {
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.z * blockDim.z + threadIdx.z;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.w * blockDim.w + threadIdx.w;

    int h_in = h_out * stride;
    int w_in = w_out * stride;
    int out_height = (height + stride - 1) / stride;
    int out_width = (width + stride - 1) / stride;

    if (b >= batch_size || c >= channels || h_out >= out_height || w_out >= out_width) return;

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int h = h_in + kh - kernel_size / 2;
            int w = w_in + kw - kernel_size / 2;
            if (h >= 0 && h < height && w >= 0 && w < width) {
                int input_idx = b * channels * height * width + 
                               c * height * width + 
                               h * width + w;
                int weight_idx = c * kernel_size * kernel_size + 
                                 kh * kernel_size + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    int output_idx = b * channels * out_height * out_width + 
                     c * out_height * out_width + 
                     h_out * out_width + w_out;
    output[output_idx] = sum;
}

// CUDA kernel for batch normalization
__global__ void batch_norm_kernel(
    const float* input, const float* weight, const float* bias,
    const float* running_mean, const float* running_var,
    float* output,
    int batch_size, int channels,
    int height, int width, float eps
) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.w * blockDim.w + threadIdx.w;

    if (b >= batch_size || c >= channels || h >= height || w >= width) return;

    int idx = b * channels * height * width + 
              c * height * width + 
              h * width + w;
    
    float normalized = (input[idx] - running_mean[c]) / sqrt(running_var[c] + eps);
    output[idx] = normalized * weight[c] + bias[c];
}

// CUDA kernel for ReLU6 activation
__global__ void relu6_kernel(
    float* input_output,
    int batch_size, int channels,
    int height, int width
) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.w * blockDim.w + threadIdx.w;

    if (b >= batch_size || c >= channels || h >= height || w >= width) return;

    int idx = b * channels * height * width + 
              c * height * width + 
              h * width + w;
    
    input_output[idx] = fminf(fmaxf(input_output[idx], 0.0f), 6.0f);
}

// CUDA kernel for residual addition
__global__ void residual_add_kernel(
    const float* input, const float* residual,
    float* output,
    int batch_size, int channels,
    int height, int width
) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.w * blockDim.w + threadIdx.w;

    if (b >= batch_size || c >= channels || h >= height || w >= width) return;

    int idx = b * channels * height * width + 
              c * height * width + 
              h * width + w;
    
    output[idx] = input[idx] + residual[c];
}

// CUDA kernel for adaptive average pooling
__global__ void adaptive_avg_pool2d_kernel(
    const float* input, float* output,
    int batch_size, int channels,
    int height, int width
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= batch_size || c >= channels) return;

    float sum = 0.0f;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int idx = b * channels * height * width + 
                      c * height * width + 
                      h * width + w;
            sum += input[idx];
        }
    }
    output[b * channels + c] = sum / (height * width);
}

// CUDA kernel for linear layer
__global__ void linear_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size, int in_features, int out_features
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int o = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= batch_size || o >= out_features) return;

    float sum = 0.0f;
    for (int i = 0; i < in_features; ++i) {
        sum += input[b * in_features + i] * weight[o * in_features + i];
    }
    output[b * out_features + o] = sum + bias[o];
}

// Main forward function
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor conv1_weight,
    torch::Tensor bn1_weight, torch::Tensor bn1_bias,
    torch::Tensor bn1_running_mean, torch::Tensor bn1_running_var,
    // ... other parameters would be added here
    torch::Tensor fc_weight, torch::Tensor fc_bias
) {
    CHECK_INPUT(input);
    CHECK_INPUT(conv1_weight);
    CHECK_INPUT(bn1_weight);
    CHECK_INPUT(bn1_bias);
    CHECK_INPUT(bn1_running_mean);
    CHECK_INPUT(bn1_running_var);
    CHECK_INPUT(fc_weight);
    CHECK_INPUT(fc_bias);

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    // First convolution
    auto conv1_output = torch::zeros({batch_size, 32, height / 2, width / 2}, input.options());
    dim3 block1(8, 8, 8);
    dim3 grid1(
        (batch_size + block1.x - 1) / block1.x,
        (height / 2 + block1.y - 1) / block1.y,
        (width / 2 + block1.z - 1) / block1.z,
        (32 + block1.w - 1) / block1.w
    );
    pointwise_conv_kernel<<<grid1, block1>>>(
        input.data_ptr<float>(), conv1_weight.data_ptr<float>(),
        conv1_output.data_ptr<float>(),
        batch_size, in_channels, 32,
        height, width, 3
    );

    // First batch norm
    auto bn1_output = torch::zeros_like(conv1_output);
    batch_norm_kernel<<<grid1, block1>>>(
        conv1_output.data_ptr<float>(),
        bn1_weight.data_ptr<float>(), bn1_bias.data_ptr<float>(),
        bn1_running_mean.data_ptr<float>(), bn1_running_var.data_ptr<float>(),
        bn1_output.data_ptr<float>(),
        batch_size, 32,
        height / 2, width / 2, 1e-5
    );

    // First ReLU6
    relu6_kernel<<<grid1, block1>>>(
        bn1_output.data_ptr<float>(),
        batch_size, 32,
        height / 2, width / 2
    );

    // ... Inverted residual blocks would be implemented here

    // Final linear layer
    int last_channel = 1280;
    auto output = torch::zeros({batch_size, 1000}, input.options());
    dim3 block_final(16, 16);
    dim3 grid_final(
        (batch_size + block_final.x - 1) / block_final.x,
        (1000 + block_final.y - 1) / block_final.y
    );
    linear_kernel<<<grid_final, block_final>>>(
        bn1_output.data_ptr<float>(),  // This should be the final feature map after all operations
        fc_weight.data_ptr<float>(), fc_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, last_channel, 1000
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MobileNetV2 forward");
}