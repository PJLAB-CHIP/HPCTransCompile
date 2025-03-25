#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <omp.h>
#include <cmath>

// Function to perform warp-level reduction
float warp_reduce_sum(float val) {
    #pragma omp parallel for reduction(+:val)
    for (int offset = 16; offset > 0; offset /= 2) {
        val += val;
    }
    return val;
}

void warp_reduce(float& sum, float& sumsq) {
    sum = warp_reduce_sum(sum);
    sumsq = warp_reduce_sum(sumsq);
}

void fused_relu_groupnorm_opt_cpu(
    float* data,
    const float* gamma,
    const float* beta,
    int N, int C, int D, int H, int W,
    int G, float eps) {
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < G; ++g) {
            const int channels_per_group = C / G;
            const int c_start = g * channels_per_group;
            const int group_size = channels_per_group * D * H * W;
            const int group_offset = n * C * D * H * W + c_start * D * H * W;

            float thread_sum = 0.f;
            float thread_sumsq = 0.f;

            #pragma omp parallel for reduction(+:thread_sum, thread_sumsq)
            for (int i = 0; i < group_size; ++i) {
                const int idx = group_offset + i;
                float val = data[idx];
                val = std::max(val, 0.f);
                data[idx] = val;
                thread_sum += val;
                thread_sumsq += val * val;
            }

            float mean = thread_sum / group_size;
            float variance = thread_sumsq / group_size - mean * mean;
            float inv_std = std::sqrt(1.0f / (variance + eps));

            #pragma omp parallel for
            for (int i = 0; i < group_size; ++i) {
                const int idx = group_offset + i;
                const int channel_idx = i / (D * H * W);
                const int c = c_start + channel_idx;
                float val = data[idx];
                val = (val - mean) * inv_std;
                val = val * gamma[c] + beta[c];
                data[idx] = val;
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_transpose,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias,
    int64_t groups,
    double eps) {

    auto y = at::conv_transpose3d(
        x,
        conv_transpose,
        /*bias=*/c10::nullopt,
        /*stride=*/{1, 1, 1},
        /*padding=*/{0, 0, 0},
        /*output_padding=*/{0, 0, 0},
        /*groups=*/1,
        /*dilation=*/{1, 1, 1}
    );

    int N = y.size(0);
    int C = y.size(1);
    int D = y.size(2);
    int H = y.size(3);
    int W = y.size(4);
    int G = groups;

    fused_relu_groupnorm_opt_cpu(
        y.data_ptr<float>(),
        group_norm_weight.data_ptr<float>(),
        group_norm_bias.data_ptr<float>(),
        N, C, D, H, W,
        G, static_cast<float>(eps)
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused ConvTranspose3D + ReLU + GroupNorm with optimized CPU operations");
}