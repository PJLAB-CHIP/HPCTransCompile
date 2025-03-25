#include <torch/extension.h>
#include <omp.h>
#include <cmath>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

template <typename scalar_t>
void fused_act_groupnorm_cpu(
    scalar_t* y,         // in/out tensor with shape [N, C]
    const scalar_t* bias,  // bias vector of length C
    const int N,
    const int C,
    const int num_groups,
    const float eps) {

    #pragma omp parallel for collapse(2)
    for (int sample = 0; sample < N; ++sample) {
        for (int group = 0; group < num_groups; ++group) {
            int channels_per_group = C / num_groups;
            int group_start = group * channels_per_group;

            // Temporary storage for sum and sum of squares
            float sum = 0.0f;
            float sum_sq = 0.0f;

            #pragma omp parallel for reduction(+:sum, sum_sq)
            for (int tid = 0; tid < channels_per_group; ++tid) {
                int channel = group_start + tid;
                int idx = sample * C + channel;
                float tmp = static_cast<float>(y[idx]) + static_cast<float>(bias[channel]);
                tmp = fminf(fmaxf(tmp, -1.0f), 1.0f); // Hardtanh activation
                float sp = log1pf(expf(tmp)); // Softplus
                float act_val = tmp * tanhf(sp); // Mish activation

                sum += act_val;
                sum_sq += act_val * act_val;
            }

            float mean = sum / channels_per_group;
            float variance = sum_sq / channels_per_group - mean * mean;
            float inv_std = rsqrtf(variance + eps);

            #pragma omp parallel for
            for (int tid = 0; tid < channels_per_group; ++tid) {
                int channel = group_start + tid;
                int idx = sample * C + channel;
                float norm_val = (y[idx] - mean) * inv_std;
                y[idx] = static_cast<scalar_t>(norm_val);
            }
        }
    }
}

torch::Tensor fused_activation_groupnorm_cpu(
    torch::Tensor y,
    torch::Tensor bias,
    int num_groups,
    double eps) {
    CHECK_CONTIGUOUS(y);
    CHECK_CONTIGUOUS(bias);
    TORCH_CHECK(y.dim() == 2, "Input tensor y must be 2D");
    int N = y.size(0);
    int C = y.size(1);
    TORCH_CHECK(C % num_groups == 0, "C must be divisible by num_groups");

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(y.scalar_type(), "fused_activation_groupnorm_cpu", ([&] {
        fused_act_groupnorm_cpu<scalar_t>(
            y.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            N,
            C,
            num_groups,
            static_cast<float>(eps));
    }));

    return y;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor weight_bias,
    torch::Tensor bias,
    int64_t num_groups,
    double eps = 1e-5) {
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(weight);
    CHECK_CONTIGUOUS(weight_bias);
    CHECK_CONTIGUOUS(bias);

    // GEMM: x @ weight.t() + weight_bias
    auto y = torch::matmul(x, weight.t()) + weight_bias;

    // Fuse second bias addition, Hardtanh, Mish, and GroupNorm into a single kernel
    y = fused_activation_groupnorm_cpu(y, bias, num_groups, eps);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused BiasAdd, Hardtanh, Mish and GroupNorm CPU forward function",
          py::arg("x"),
          py::arg("weight"),
          py::arg("weight_bias"),
          py::arg("bias"),
          py::arg("num_groups"),
          py::arg("eps") = 1e-5);
}