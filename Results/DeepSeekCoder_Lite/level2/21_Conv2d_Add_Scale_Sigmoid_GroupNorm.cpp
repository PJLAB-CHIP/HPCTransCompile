#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <cmath>
#include <omp.h>

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cpu(), #x " must be a CPU tensor.")

// Optimized kernel: leverages shared memory for frequently accessed data
void shared_memory_coalesced_access_kernel(
    const at::TensorAccessor<float, 4> x,    // input tensor (result of conv2d), shape [N, C, H, W]
    at::TensorAccessor<float, 4> y,          // output tensor, same shape
    const at::TensorAccessor<float, 1> bias, // bias for elementwise op (either size 1 or C)
    const at::TensorAccessor<float, 1> scale,// scale for elementwise op (either size 1 or C)
    const at::TensorAccessor<float, 1> gn_weight, // group norm weight, shape [C]
    const at::TensorAccessor<float, 1> gn_bias,   // group norm bias, shape [C]
    int N, int C, int H, int W,
    int num_groups,
    bool bias_broadcast,
    bool scale_broadcast,
    float eps) {

    int channels_per_group = C / num_groups;

    #pragma omp parallel for
    for (int sample_idx = 0; sample_idx < N; ++sample_idx) {
        for (int group_idx = 0; group_idx < num_groups; ++group_idx) {
            for (int thread_idx = 0; thread_idx < omp_get_max_threads(); ++thread_idx) {
                int channels_per_thread = (channels_per_group + omp_get_max_threads() - 1) / omp_get_max_threads();
                int start_channel = channels_per_group * thread_idx;
                int end_channel = std::min(start_channel + channels_per_thread, C);

                for (int c = start_channel; c < end_channel; ++c) {
                    int group_channel_offset = group_idx * channels_per_group;
                    int group_size = channels_per_group * H * W;
                    int sample_offset = sample_idx * C * H * W;
                    int idx = sample_offset + c * (H * W);

                    float local_sum = 0.0f;
                    float local_sum_sq = 0.0f;

                    for (int i = 0; i < group_size; ++i) {
                        int hw = i % (H * W);
                        float in_val = x[sample_idx][c][i / (H * W)][hw];
                        float b_val = bias_broadcast ? bias[0] : bias[c];
                        float s_val = scale_broadcast ? scale[0] : scale[c];
                        float pre_act = (in_val + b_val) * s_val;
                        float v = 1.0f / (1.0f + expf(-pre_act));  // sigmoid activation

                        y[sample_idx][c][i / (H * W)][hw] = v;
                        local_sum += v;
                        local_sum_sq += v * v;
                    }

                    float group_mean = local_sum / group_size;
                    float group_var = local_sum_sq / group_size - group_mean * group_mean;
                    float inv_std = 1.0f / sqrtf(group_var + eps);

                    for (int i = 0; i < group_size; ++i) {
                        int hw = i % (H * W);
                        float v = y[sample_idx][c][i / (H * W)][hw];
                        float normalized = (v - group_mean) * inv_std;
                        float gamma = gn_weight[c];
                        float beta = gn_bias[c];
                        y[sample_idx][c][i / (H * W)][hw] = gamma * normalized + beta;
                    }
                }
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

    CHECK_INPUT(x);
    CHECK_INPUT(conv_weight);
    if (conv_bias.defined()) CHECK_INPUT(conv_bias);
    CHECK_INPUT(bias);
    CHECK_INPUT(scale);
    CHECK_INPUT(gn_weight);
    CHECK_INPUT(gn_bias);

    x = at::conv2d(x, conv_weight, conv_bias);

    at::Tensor y = at::empty_like(x);

    bool bias_broadcast = (bias.numel() == 1);
    bool scale_broadcast = (scale.numel() == 1);

    float eps = 1e-5;

    shared_memory_coalesced_access_kernel(x, y, bias, scale, gn_weight, gn_bias,
                                           num_groups, bias_broadcast, scale_broadcast, eps);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Shared memory coalesced access kernel (CPU)");
}