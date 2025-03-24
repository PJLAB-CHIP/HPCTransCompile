#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <cfloat>

// Fused kernel: Applies ReLU and then performs 2D max pooling in one pass.
// Workload is evenly distributed using a grid-stride loop.
__global__ void fused_relu_pool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch, int channels,
    int height, int width,
    int pool_h, int pool_w, int stride
) {
    int out_h = (height - pool_h) / stride + 1;
    int out_w = (width - pool_w) / stride + 1;
    int total = batch * channels * out_h * out_w;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int tmp = idx;
        int w = tmp % out_w; tmp /= out_w;
        int h = tmp % out_h; tmp /= out_h;
        int c = tmp % channels; tmp /= channels;
        int b = tmp;

        int in_row_start = h * stride;
        int in_col_start = w * stride;
        // Initialize to 0 since with ReLU negatives become 0.
        float max_val = 0.0f;

        for (int i = 0; i < pool_h; i++) {
            for (int j = 0; j < pool_w; j++) {
                int in_row = in_row_start + i;
                int in_col = in_col_start + j;
                float val = input[((b * channels + c) * height + in_row) * width + in_col];
                // Apply ReLU inline
                float relu_val = fmaxf(val, 0.0f);
                if (relu_val > max_val) {
                    max_val = relu_val;
                }
            }
        }
        output[idx] = max_val;
    }
}

// Simple flattening kernel using a grid-stride loop
__global__ void flatten_kernel(const float* __restrict__ input, float* __restrict__ output, int total) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        output[idx] = input[idx];
    }
}

// Forward function for the LeNet-5 network that uses the fused ReLU+Pool kernel
// to better distribute workloads evenly and reduce kernel launch overhead.

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv1_weight, torch::Tensor conv1_bias,
    torch::Tensor conv2_weight, torch::Tensor conv2_bias,
    torch::Tensor fc1_weight, torch::Tensor fc1_bias,
    torch::Tensor fc2_weight, torch::Tensor fc2_bias,
    torch::Tensor fc3_weight, torch::Tensor fc3_bias
) {
    // Move all inputs to CUDA
    x = x.to(torch::kCUDA);
    conv1_weight = conv1_weight.to(torch::kCUDA);
    conv1_bias = conv1_bias.to(torch::kCUDA);
    conv2_weight = conv2_weight.to(torch::kCUDA);
    conv2_bias = conv2_bias.to(torch::kCUDA);
    fc1_weight = fc1_weight.to(torch::kCUDA);
    fc1_bias = fc1_bias.to(torch::kCUDA);
    fc2_weight = fc2_weight.to(torch::kCUDA);
    fc2_bias = fc2_bias.to(torch::kCUDA);
    fc3_weight = fc3_weight.to(torch::kCUDA);
    fc3_bias = fc3_bias.to(torch::kCUDA);

    // First Convolutional Layer
    auto conv1 = torch::conv2d(x, conv1_weight, conv1_bias, {1, 1});

    // Instead of launching separate ReLU and max_pool kernels, we fuse them.
    int B = conv1.size(0);
    int C = conv1.size(1);
    int H = conv1.size(2);
    int W = conv1.size(3);
    int pool_h = 2, pool_w = 2, stride = 2;
    int out_h = (H - pool_h) / stride + 1;
    int out_w = (W - pool_w) / stride + 1;

    auto pool1 = torch::empty({B, C, out_h, out_w}, conv1.options());
    int total_pool1 = B * C * out_h * out_w;
    int threads = 256;
    int blocks = (total_pool1 + threads - 1) / threads;
    fused_relu_pool_kernel<<<blocks, threads>>>(
        conv1.data_ptr<float>(), pool1.data_ptr<float>(), B, C, H, W, pool_h, pool_w, stride);

    // Second Convolutional Layer
    auto conv2 = torch::conv2d(pool1, conv2_weight, conv2_bias, {1, 1});
    B = conv2.size(0);
    C = conv2.size(1);
    H = conv2.size(2);
    W = conv2.size(3);
    out_h = (H - pool_h) / stride + 1;
    out_w = (W - pool_w) / stride + 1;
    auto pool2 = torch::empty({B, C, out_h, out_w}, conv2.options());
    int total_pool2 = B * C * out_h * out_w;
    blocks = (total_pool2 + threads - 1) / threads;
    fused_relu_pool_kernel<<<blocks, threads>>>(
        conv2.data_ptr<float>(), pool2.data_ptr<float>(), B, C, H, W, pool_h, pool_w, stride);

    // Flatten the output
    auto flat = pool2.view({pool2.size(0), -1});

    // Fully connected layers are computed using torch::linear which are highly optimized (e.g., via cuBLAS)
    auto fc1 = torch::linear(flat, fc1_weight, fc1_bias);
    fc1 = torch::relu(fc1);
    auto fc2 = torch::linear(fc1, fc2_weight, fc2_bias);
    fc2 = torch::relu(fc2);
    auto fc3 = torch::linear(fc2, fc3_weight, fc3_bias);

    return fc3;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LeNet-5 forward pass with fused ReLU and pooling");
}
