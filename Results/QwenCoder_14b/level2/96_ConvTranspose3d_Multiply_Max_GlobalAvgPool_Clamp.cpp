#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <omp.h>
#include <algorithm>
#include <cmath>

template<int POOL_K>
void fused_pooling_cpu(
    const float* input,
    float* output,
    const int N, const int C, const int D, const int H, const int W,
    const float scale) {

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const int D_pool = D / POOL_K;
            const int H_pool = H / POOL_K;
            const int W_pool = W / POOL_K;
            const int total_windows = D_pool * H_pool * W_pool;

            const int channel_offset = ((n * C + c) * D * H * W);
            float local_sum = 0.0f;

            for (int win_idx = 0; win_idx < total_windows; ++win_idx) {
                const int d_pool_idx = win_idx / (H_pool * W_pool);
                const int rem = win_idx % (H_pool * W_pool);
                const int h_pool_idx = rem / W_pool;
                const int w_pool_idx = rem % W_pool;

                const int d_start = d_pool_idx * POOL_K;
                const int h_start = h_pool_idx * POOL_K;
                const int w_start = w_pool_idx * POOL_K;

                float max_val = -FLT_MAX;

                for (int i = d_start; i < d_start + POOL_K; ++i) {
                    for (int j = h_start; j < h_start + POOL_K; ++j) {
                        for (int k = w_start; k < w_start + POOL_K; ++k) {
                            const int index = channel_offset + (i * H * W) + (j * W) + k;
                            const float val = input[index] * scale;
                            max_val = std::max(max_val, val);
                        }
                    }
                }
                local_sum += max_val;
            }

            float avg = local_sum / total_windows;
            avg = std::min(std::max(avg, 0.0f), 1.0f); // Clamps to [0,1] range
            output[n * C + c] = avg;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    double scale,
    int64_t maxpool_kernel_size,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias) {

    auto conv_out = torch::conv_transpose3d(
        x, conv_transpose, conv_transpose_bias,
        {stride, stride, stride},
        {padding, padding, padding}
    );

    conv_out = conv_out.contiguous();
    const int N = conv_out.size(0);
    const int C = conv_out.size(1);
    const int D = conv_out.size(2);
    const int H = conv_out.size(3);
    const int W = conv_out.size(4);

    auto output = torch::empty({N, C}, conv_out.options());

    if (maxpool_kernel_size == 2) {
        fused_pooling_cpu<2>(
            conv_out.data_ptr<float>(), output.data_ptr<float>(),
            N, C, D, H, W, static_cast<float>(scale));
    } else {
        fused_pooling_cpu<4>(
            conv_out.data_ptr<float>(), output.data_ptr<float>(),
            N, C, D, H, W, static_cast<float>(scale));
    }

    return output.view({N, C, 1, 1, 1});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp CPU kernel with OpenMP parallelization");
}