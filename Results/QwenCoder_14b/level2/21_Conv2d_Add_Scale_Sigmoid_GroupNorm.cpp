#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <omp.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")

// Function to perform the sigmoid activation
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// CPU implementation of the shared memory optimized kernel
void shared_memory_coalesced_access_cpu(
    const float* x,    // input tensor (result of conv2d), shape [N, C, H, W]
    float* y,          // output tensor, same shape
    const float* bias, // bias for elementwise op (either size 1 or C)
    const float* scale, // scale for elementwise op (either size 1 or C)
    const float* gn_weight, // group norm weight, shape [C]
    const float* gn_bias,   // group norm bias, shape [C]
    int N, int C, int H, int W,
    int num_groups,
    bool bias_broadcast,
    bool scale_broadcast,
    float eps) {

    #pragma omp parallel for collapse(2)
    for (int sample_idx = 0; sample_idx < N; ++sample_idx) {
        for (int group_idx = 0; group_idx < num_groups; ++group_idx) {
            int channels_per_group = C / num_groups;
            int group_size = channels_per_group * H * W;

            int sample_offset = sample_idx * C * H * W;
            int group_channel_offset = group_idx * channels_per_group;

            float local_sum = 0.0f;
            float local_sum_sq = 0.0f;

            for (int i = 0; i < group_size; ++i) {
                int c_local = i / (H * W);
                int hw = i % (H * W);
                int c = group_channel_offset + c_local;
                int idx = sample_offset + c * (H * W) + hw;

                float in_val = x[idx];
                float b_val = bias_broadcast ? bias[0] : bias[c];
                float s_val = scale_broadcast ? scale[0] : scale[c];
                float pre_act = (in_val + b_val) * s_val;
                float v = sigmoid(pre_act);  // sigmoid activation

                y[idx] = v;
                local_sum += v;
                local_sum_sq += v * v;
            }

            float group_mean = local_sum / group_size;
            float group_var = local_sum_sq / group_size - group_mean * group_mean;
            float inv_std = 1.0f / std::sqrt(group_var + eps);

            for (int i = 0; i < group_size; ++i) {
                int c_local = i / (H * W);
                int hw = i % (H * W);
                int c = group_channel_offset + c_local;
                int idx = sample_offset + c * (H * W) + hw;

                float v = y[idx];
                float normalized = (v - group_mean) * inv_std;
                float gamma = gn_weight[c];
                float beta = gn_bias[c];
                y[idx] = gamma * normalized + beta;
            }
        }
    }
}

// Forward function
at::Tensor module_fn_forward(
    at::Tensor x,
    at::Tensor conv_weight,
    at::Tensor conv_bias,
    at::Tensor bias,
    at::Tensor scale,
    at::Tensor gn_weight,
    at::Tensor gn_bias,
    int64_t num_groups) {

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(conv_weight);
    if (conv_bias.defined()) CHECK_CONTIGUOUS(conv_bias);
    CHECK_CONTIGUOUS(bias);
    CHECK_CONTIGUOUS(scale);
    CHECK_CONTIGUOUS(gn_weight);
    CHECK_CONTIGUOUS(gn_bias);

    x = at::conv2d(x, conv_weight, conv_bias);

    at::Tensor y = at::empty_like(x);

    bool bias_broadcast = (bias.numel() == 1);
    bool scale_broadcast = (scale.numel() == 1);

    float eps = 1e-5;

    shared_memory_coalesced_access_cpu(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        x.size(0), x.size(1), x.size(2), x.size(3),
        num_groups, bias_broadcast, scale_broadcast, eps);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Shared memory coalesced access kernel (CPU)");
}