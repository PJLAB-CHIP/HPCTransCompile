#include <torch/extension.h>
#include <vector>
#include <cfloat>
#include <omp.h>

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

    #pragma omp parallel for
    for (int idx = 0; idx < N * D * H * W; ++idx) {
        // Calculate total elements and stride for thread processing
        const int w = idx % W;
        int temp = idx / W;
        const int h = temp % H;
        temp /= H;
        const int d = temp % D;
        const int n = temp / D;

        // First pass: find maximum value across channels
        float max_val = -FLT_MAX;
        for (int c = 0; c < C; ++c) {
            const int input_idx = n * (C * D * H * W) + c * (D * H * W) + d * (H * W) + h * W + w;
            max_val = std::fmax(max_val, pool_result[n][c][d][h][w].item<float>());
        }

        // Second pass: compute sum of exponentials
        float sum_exp = 0.0f;
        for (int c = 0; c < C; ++c) {
            const int input_idx = n * (C * D * H * W) + c * (D * H * W) + d * (H * W) + h * W + w;
            sum_exp += std::exp(pool_result[n][c][d][h][w].item<float>() - max_val);
        }

        // Compute final result with ReLU
        float result = max_val + std::log(sum_exp);
        result = std::fmax(0.0f, result);
        
        // Write to output
        output[n][0][d][h][w] = result;
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided fused logsumexp and ReLU kernel");
}