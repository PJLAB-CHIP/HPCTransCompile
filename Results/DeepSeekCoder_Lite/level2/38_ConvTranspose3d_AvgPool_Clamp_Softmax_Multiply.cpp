#include <torch/extension.h>
#include <vector>
#include <cfloat>
#include <cmath>
#include <omp.h>

__forceinline__ float clamp_value(float val, float min_val, float max_val) {
    return std::fmin(std::fmax(val, min_val), max_val);
}

__forceinline__ int compute_offset(int n, int c, int d, int h, int w, int C, int D, int H, int W) {
    return n * (C * D * H * W) + c * (D * H * W) + d * (H * W) + h * W + w;
}

__forceinline__ void compute_coordinates(int pos, int D, int H, int W, int& n, int& d, int& h, int& w) {
    const int DHW = D * H * W;
    const int HW = H * W;
    n = pos / DHW;
    int rem = pos % DHW;
    d = rem / HW;
    rem = rem % HW;
    h = rem / W;
    w = rem % W;
}

__forceinline__ float compute_max_value(const float* input, int n, int d, int h, int w, int C, int D, int H, int W, float clamp_min, float clamp_max) {
    float max_val = -FLT_MAX;
    for (int c = 0; c < C; ++c) {
        const int offset = compute_offset(n, c, d, h, w, C, D, H, W);
        float val = clamp_value(input[offset], clamp_min, clamp_max);
        max_val = std::fmax(max_val, val);
    }
    return max_val;
}

__forceinline__ float compute_sum_exp(const float* input, int n, int d, int h, int w, int C, int D, int H, int W, float max_val, float clamp_min, float clamp_max) {
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        const int offset = compute_offset(n, c, d, h, w, C, D, H, W);
        float val = clamp_value(input[offset], clamp_min, clamp_max);
        sum += std::exp(val - max_val);
    }
    return sum;
}

__forceinline__ void compute_softmax_and_multiply(const float* input, float* output, int n, int d, int h, int w, int C, int D, int H, int W, float max_val, float inv_sum, float clamp_min, float clamp_max) {
    for (int c = 0; c < C; ++c) {
        const int offset = compute_offset(n, c, d, h, w, C, D, H, W);
        float val = clamp_value(input[offset], clamp_min, clamp_max);
        output[offset] = 2.0f * std::exp(val - max_val) * inv_sum;
    }
}

torch::Tensor module_fn(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t pool_kernel_size,
    double clamp_min,
    double clamp_max,
    torch::Tensor conv_transpose_weight,
    torch::Tensor conv_transpose_bias
) {
    auto device = x.device();
    conv_transpose_weight = conv_transpose_weight.to(device);
    conv_transpose_bias = conv_transpose_bias.to(device);

    std::vector<int64_t> stride_vec(3, stride);
    std::vector<int64_t> padding_vec(3, padding);
    std::vector<int64_t> output_padding_vec(3, output_padding);
    std::vector<int64_t> pool_kernel_size_vec(3, pool_kernel_size);

    auto conv_out = torch::conv_transpose3d(
        x, conv_transpose_weight, conv_transpose_bias,
        stride_vec, padding_vec, output_padding_vec
    );

    auto pool_out = torch::avg_pool3d(conv_out, pool_kernel_size_vec);
    auto result = torch::empty_like(pool_out);

    const int N = pool_out.size(0);
    const int C = pool_out.size(1);
    const int D = pool_out.size(2);
    const int H = pool_out.size(3);
    const int W = pool_out.size(4);

    const int threads = 128;  // Reduced thread count for potentially better occupancy
    const int blocks = std::min((N * D * H * W + threads - 1) / threads, 65535);
    const int shared_mem_size = 2 * threads * sizeof(float); // For max_values and sum_exp

    #pragma omp parallel for
    for (int pos = 0; pos < N * D * H * W; ++pos) {
        int n, d, h, w;
        compute_coordinates(pos, D, H, W, n, d, h, w);

        // First pass: find max value
        float max_val = compute_max_value(pool_out.data_ptr<float>(), n, d, h, w, C, D, H, W, static_cast<float>(clamp_min), static_cast<float>(clamp_max));

        // Second pass: compute sum of exponentials
        float sum = compute_sum_exp(pool_out.data_ptr<float>(), n, d, h, w, C, D, H, W, max_val, static_cast<float>(clamp_min), static_cast<float>(clamp_max));

        // Third pass: compute softmax and multiply by 2
        const float inv_sum = 1.0f / sum;
        compute_softmax_and_multiply(pool_out.data_ptr<float>(), result.data_ptr<float>(), n, d, h, w, C, D, H, W, max_val, inv_sum, static_cast<float>(clamp_min), static_cast<float>(clamp_max));
    }

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Modular kernel with device functions");
}