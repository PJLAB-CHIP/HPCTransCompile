#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cfloat>

__global__ void strided_fused_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N, const int C, const int D, const int H, const int W) {
    
    // Calculate total elements and stride for thread processing
    const int total_elements = N * D * H * W;
    const int num_threads = blockDim.x * gridDim.x;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = D * H * W;
    
    // Each thread processes multiple elements using stride loop
    for (int idx = thread_id; idx < total_elements; idx += num_threads) {
        // Decode indices
        const int w = idx % W;
        int temp = idx / W;
        const int h = temp % H;
        temp /= H;
        const int d = temp % D;
        const int n = temp / D;

        // First pass: find maximum value across channels
        float max_val = -FLT_MAX;
        #pragma unroll 4
        for (int c = 0; c < C; ++c) {
            const int input_idx = n * (C * stride) + c * stride + d * (H * W) + h * W + w;
            max_val = fmaxf(max_val, input[input_idx]);
        }

        // Second pass: compute sum of exponentials
        float sum_exp = 0.0f;
        #pragma unroll 4
        for (int c = 0; c < C; ++c) {
            const int input_idx = n * (C * stride) + c * stride + d * (H * W) + h * W + w;
            sum_exp += expf(input[input_idx] - max_val);
        }

        // Compute final result with ReLU
        float result = max_val + logf(sum_exp);
        result = fmaxf(0.0f, result);
        
        // Write to output
        output[idx] = result;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias) {

    // Perform 3D convolution using PyTorch
    auto conv_result = torch::conv3d(x, conv_weight, conv_bias, 
                                   {stride, stride, stride}, 
                                   {padding, padding, padding});

    // Perform max pooling using PyTorch
    auto pool_result = torch::max_pool3d(conv_result, {2, 2, 2}, {2, 2, 2});

    // Get dimensions for the fused logsumexp and ReLU operations
    const int N = pool_result.size(0);
    const int C = pool_result.size(1);
    const int D = pool_result.size(2);
    const int H = pool_result.size(3);
    const int W = pool_result.size(4);

    // Create output tensor
    auto output = torch::empty({N, 1, D, H, W}, pool_result.options());

    // Launch kernel with stride-based processing
    const int block_size = 256;
    const int num_blocks = (N * D * H * W + block_size - 1) / block_size;
    
    strided_fused_kernel<<<num_blocks, block_size>>>(
        pool_result.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided fused logsumexp and ReLU kernel");
}