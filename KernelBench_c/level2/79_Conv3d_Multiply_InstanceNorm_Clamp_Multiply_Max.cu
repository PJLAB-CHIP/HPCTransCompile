#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel声明
__global__ void multiply_kernel(float* input, const float* multiplier, 
    int num_elements, int channels, int depth, int height, int width);
__global__ void clamp_kernel(float* input, float min_val, float max_val, 
    int num_elements);
__global__ void max_reduce_kernel(const float* input, float* output, 
    int batch_size, int channels, int depth, int height, int width);

// 对齐Python Model参数的forward函数
torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor multiplier,
    float clamp_min,
    float clamp_max) {
    
    // 输入检查
    CHECK_INPUT(input);
    CHECK_INPUT(conv_weight);
    CHECK_INPUT(conv_bias);
    CHECK_INPUT(multiplier);

    // 执行3D卷积
    auto x = torch::conv3d(input, conv_weight, conv_bias);

    // 获取张量维度
    int batch_size = x.size(0);
    int channels = x.size(1);
    int depth = x.size(2);
    int height = x.size(3);
    int width = x.size(4);
    int num_elements = x.numel();

    // 第一次乘法（修复通道索引计算）
    {
        dim3 block(256);
        dim3 grid((num_elements + block.x - 1) / block.x);
        multiply_kernel<<<grid, block>>>(
            x.data_ptr<float>(),
            multiplier.data_ptr<float>(),
            num_elements,
            channels,
            depth,
            height,
            width);
        cudaDeviceSynchronize();
    }

    // 实例归一化（与PyTorch严格一致）
    x = torch::instance_norm(x, 
        /*running_mean=*/torch::Tensor(),
        /*running_var=*/torch::Tensor(),
        /*weight=*/torch::Tensor(),
        /*bias=*/torch::Tensor(),
        /*use_input_stats=*/true,
        /*momentum=*/0.0,
        /*eps=*/1e-5,
        /*cudnn_enabled=*/false);

    // 截断操作
    {
        dim3 block(256);
        dim3 grid((num_elements + block.x - 1) / block.x);
        clamp_kernel<<<grid, block>>>(
            x.data_ptr<float>(),
            clamp_min,
            clamp_max,
            num_elements);
        cudaDeviceSynchronize();
    }

    // 第二次乘法
    {
        dim3 block(256);
        dim3 grid((num_elements + block.x - 1) / block.x);
        multiply_kernel<<<grid, block>>>(
            x.data_ptr<float>(),
            multiplier.data_ptr<float>(),
            num_elements,
            channels,
            depth,
            height,
            width);
        cudaDeviceSynchronize();
    }

    // 沿通道维度取最大值（修复输出维度）
    auto output = torch::empty({batch_size, 1, depth, height, width}, x.options());
    {
        dim3 block(16, 16);
        dim3 grid(
            (depth + block.x - 1) / block.x,
            (height + block.y - 1) / block.y,
            batch_size * width);
        max_reduce_kernel<<<grid, block>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            channels,
            depth,
            height,
            width);
        cudaDeviceSynchronize();
    }

    return output;
}

// 核函数实现（关键修正）
__global__ void multiply_kernel(float* input, const float* multiplier, 
    int num_elements, int channels, int depth, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // 正确的通道索引计算
        int elements_per_channel = depth * height * width;
        int c = (idx / elements_per_channel) % channels;
        input[idx] *= multiplier[c];
    }
}

__global__ void clamp_kernel(float* input, float min_val, float max_val, 
    int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        input[idx] = fminf(fmaxf(input[idx], min_val), max_val);
    }
}

__global__ void max_reduce_kernel(const float* input, float* output, 
    int batch_size, int channels, int depth, int height, int width) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int bw = blockIdx.z;
    
    // 分解批次和宽度维度
    int b = bw / width;
    int w = bw % width;

    if (d >= depth || h >= height || b >= batch_size || w >= width) return;

    float max_val = -INFINITY;
    for (int c = 0; c < channels; ++c) {
        // 输入索引计算
        int idx = (((b * channels + c) * depth + d) * height + h) * width + w;
        max_val = fmaxf(max_val, input[idx]);
    }
    // 输出索引修正（保持通道维度）
    int out_idx = (((b * 1) * depth + d) * height + h) * width + w;
    output[out_idx] = max_val;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Forward pass of the optimized model");
}