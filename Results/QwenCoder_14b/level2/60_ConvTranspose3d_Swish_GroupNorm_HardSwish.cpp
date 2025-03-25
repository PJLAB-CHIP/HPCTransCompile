#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <omp.h>
#include <cmath>

namespace py = pybind11;

template<typename T>
T warp_reduce_sum(T val) {
    #pragma omp parallel for reduction(+:val)
    for (int offset = 16; offset > 0; offset /= 2) {
        val += val;
    }
    return val;
}

void warp_reduce_double(double& sum, double& sumsq) {
    #pragma omp parallel for reduction(+:sum) reduction(+:sumsq)
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += sum;
        sumsq += sumsq;
    }
}

void warp_optimized_cpu(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    int N, int C, int D, int H, int W,
    int groups,
    float eps
) {
    const int channels_per_group = C / groups;
    const int group_size = channels_per_group * D * H * W;

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups; ++g) {
            const int base = n * (C * D * H * W) + g * group_size;

            double local_sum = 0.0;
            double local_sumsq = 0.0;

            for (int i = 0; i < group_size; ++i) {
                float x = input[base + i];
                float sw = x / (1.0f + expf(-x));
                local_sum += sw;
                local_sumsq += sw * sw;
            }

            warp_reduce_double(local_sum, local_sumsq);

            double mean = local_sum / group_size;
            double variance = local_sumsq / group_size - mean * mean;
            double inv_std = sqrtf(1.0f / (variance + eps));

            for (int i = 0; i < group_size; ++i) {
                float x = input[base + i];
                float sw = x / (1.0f + expf(-x));
                const int c = i / (D * H * W);
                const int gc = g * channels_per_group + c;
                float norm = (sw - mean) * inv_std;
                float y = norm * gamma[gc] + beta[gc];
                output[base + i] = y * fminf(fmaxf(y + 3.0f, 0.0f), 6.0f) / 6.0f;
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    int stride,
    int padding,
    int groups,
    float eps,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias
) {
    x = torch::conv_transpose3d(x, conv_transpose, conv_transpose_bias, stride, padding);
    torch::Tensor output = torch::empty_like(x);

    warp_optimized_cpu(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        group_norm_weight.data_ptr<float>(),
        group_norm_bias.data_ptr<float>(),
        x.size(0), x.size(1), x.size(2), x.size(3), x.size(4),
        groups,
        eps
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized fused kernel");
}