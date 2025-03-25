#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

void strided_fused_cpu(
    const float* input,
    float* output,
    const int N, const int C, const int D, const int H, const int W) {
    
    const int total_elements = N * D * H * W;
    const int num_threads = omp_get_max_threads();
    
    #pragma omp parallel for
    for (int idx = 0; idx < total_elements; idx++) {
        const int w = idx % W;
        int temp = idx / W;
        const int h = temp % H;
        temp /= H;
        const int d = temp % D;
        const int n = temp / D;

        float max_val = -FLT_MAX;
        for (int c = 0; c < C; ++c) {
            const int input_idx = n * (C * D * H * W) + c * (D * H * W) + d * (H * W) + h * W + w;
            max_val = std::max(max_val, input[input_idx]);
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < C; ++c) {
            const int input_idx = n * (C * D * H * W) + c * (D * H * W) + d * (H * W) + h * W + w;
            sum_exp += std::exp(input[input_idx] - max_val);
        }

        float result = max_val + std::log(sum_exp);
        result = std::max(0.0f, result);
        
        output[idx] = result;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias) {

    auto conv_result = torch::conv3d(x, conv_weight, conv_bias, 
                                   {stride, stride, stride}, 
                                   {padding, padding, padding});

    auto pool_result = torch::max_pool3d(conv_result, {2, 2, 2}, {2, 2, 2});

    const int N = pool_result.size(0);
    const int C = pool_result.size(1);
    const int D = pool_result.size(2);
    const int H = pool_result.size(3);
    const int W = pool_result.size(4);

    auto output = torch::empty({N, 1, D, H, W}, pool_result.options());

    strided_fused_cpu(
        pool_result.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided fused logsumexp and ReLU kernel");
}