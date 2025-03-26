```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Helper function to get threads per block
inline int getNumThreads(int n) {
    const int threads = 512;
    return (n < threads) ? nextPow2(n) : threads;
}

// Helper function to get number of blocks
inline int getNumBlocks(int n, int threads) {
    return (n + threads - 1) / threads;
}

// Kernel for window partition
__global__ void window_partition_kernel(
    const float* input, float* output,
    int B, int H, int W, int C, int window_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * (H/window_size) * (W/window_size) * window_size * window_size * C) return;

    int b = idx / ((H/window_size) * (W/window_size) * window_size * window_size * C);
    int tmp = idx % ((H/window_size) * (W/window_size) * window_size * window_size * C);
    int h1 = tmp / ((W/window_size) * window_size * window_size * C);
    tmp = tmp % ((W/window_size) * window_size * window_size * C);
    int w1 = tmp / (window_size * window_size * C);
    tmp = tmp % (window_size * window_size * C);
    int h2 = tmp / (window_size * C);
    tmp = tmp % (window_size * C);
    int w2 = tmp / C;
    int c = tmp % C;

    int input_h = h1 * window_size + h2;
    int input_w = w1 * window_size + w2;
    int input_idx = ((b * H + input_h) * W + input_w) * C + c;
    
    output[idx] = input[input_idx];
}

torch::Tensor window_partition_cuda(torch::Tensor x, int window_size) {
    CHECK_INPUT(x);
    
    auto B = x.size(0);
    auto H = x.size(1);
    auto W = x.size(2);
    auto C = x.size(3);
    
    auto output = torch::empty({B * (H/window_size) * (W/window_size), window_size, window_size, C}, 
                              x.options());
    
    int threads = getNumThreads(B * (H/window_size) * (W/window_size) * window_size * window_size * C);
    int blocks = getNumBlocks(B * (H/window_size) * (W/window_size) * window_size * window_size * C, threads);
    
    window_partition_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), output.data_ptr<float>(),
        B, H, W, C, window_size
    );
    
    return output;
}

// Kernel for window reverse
__global__ void window_reverse_kernel(
    const float* input, float* output,
    int B, int H, int W, int C, int window_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * W * C) return;

    int b = idx / (H * W * C);
    int tmp = idx % (H * W * C);
    int h = tmp / (W * C);
    tmp = tmp % (W * C);
    int w = tmp / C;
    int c = tmp % C;

    int h1 = h / window_size;
    int h2 = h % window_size;
    int w1 = w / window_size;
    int w2 = w % window_size;
    
    int input_idx = ((b * (H/window_size) * (W/window_size) + h1 * (W/window_size) + w1) * window_size * window_size + h2 * window_size + w2) * C + c;
    
    output[idx] = input[input_idx];
}

torch::Tensor window_reverse_cuda(torch::Tensor windows, int window_size, int H, int W) {
    CHECK_INPUT(windows);
    
    auto B = windows.size(0) / (H * W / window_size / window_size);
    auto C = windows.size(3);
    
    auto output = torch::empty({B, H, W, C}, windows.options());
    
    int threads = getNumThreads(B * H * W * C);
    int blocks = getNumBlocks(B * H * W * C, threads);
    
    window_reverse_kernel<<<blocks, threads>>>(
        windows.data_ptr<float>(), output.data_ptr<float>(),
        B, H, W, C, window_size
    );
    
    return output;
}

// Kernel for MLP
__global__ void mlp_kernel(
    const float* x, const float* fc1_weight, const float* fc1_bias,
    const float* fc2_weight, const float* fc2_bias, float* output,
    int B, int N, int in_features, int hidden_features, int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N * out_features) return;

    int b = idx / (N * out_features);
    int n = (idx % (N * out_features)) / out_features;
    int o = idx % out_features;

    // First linear layer
    float val = 0.0f;
    for (int i = 0; i < in_features; i++) {
        val += x[(b * N + n) * in_features + i] * fc1_weight[o * in_features + i];
    }
    val += fc1_bias[o];
    
    // GELU activation
    val = 0.5f * val * (1.0f + tanhf(0.7978845608028654f * (val + 0.044715f * val * val * val)));
    
    // Second linear layer
    float final_val = 0.0f;
    for (int i = 0; i < hidden_features; i++) {
        final_val += val * fc2_weight[o * hidden_features + i];
    }
    final_val += fc2_bias[o];
    
    output[idx] = final_val;
}

torch::Tensor mlp_cuda(
    torch::Tensor x, torch::Tensor fc1_weight, torch::Tensor fc1_bias,
    torch::Tensor fc2_weight, torch::Tensor fc2_bias
) {
    CHECK_INPUT(x);
    CHECK_INPUT(fc1_weight);
    CHECK_INPUT(fc1_bias);
    CHECK_INPUT(fc2_weight);
    CHECK_INPUT(fc2_bias);
    
    auto B = x.size(0);
    auto N = x.size(1);
    auto in_features = x.size(2);
    auto hidden_features = fc1_weight.size(0);
    auto out_features = fc2_weight.size(0);
    
    auto output = torch::empty({B, N, out_features}, x.options());
    
    int threads = getNumThreads(B * N * out_features);
    int blocks = getNumBlocks(B * N * out_features, threads);
    
    mlp_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), fc1_weight.data_ptr<float>(), fc1_bias.data_ptr<float>(),
        fc2_weight.data_ptr<float>(), fc2_bias.data_ptr<float>(), output.data_ptr<float>(),
        B, N, in_features, hidden_features, out_features
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("window_partition", &window_partition_cuda, "Window partition (CUDA)");
    m.def("window_reverse", &window_reverse_cuda, "Window reverse (CUDA)");
    m.def("mlp", &mlp_cuda, "MLP (CUDA)");
}
```