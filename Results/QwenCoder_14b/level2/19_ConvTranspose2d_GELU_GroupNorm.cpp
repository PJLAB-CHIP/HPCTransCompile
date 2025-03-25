#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cmath>
#include <omp.h>

// Function to compute GELU activation
float gelu(float v) {
    return 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
}

// CPU implementation of the fused GELU and Group Normalization
void fused_gelu_group_norm_cpu(
    const float* in,
    float* out,
    int group_size,
    int hw,
    int channels_per_group,
    int C,
    int num_groups,
    float eps,
    const float* gn_weight,
    const float* gn_bias) {

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < num_groups; ++n) {
        for (int g = 0; g < num_groups; ++g) {
            int base = n * C * hw + g * channels_per_group * hw;
            float local_sum = 0.0f;
            float local_sum_sq = 0.0f;

            // Apply GELU activation
            for (int idx = 0; idx < group_size; ++idx) {
                float v = in[base + idx];
                float gelu_val = gelu(v);
                out[base + idx] = gelu_val;
                local_sum += gelu_val;
                local_sum_sq += gelu_val * gelu_val;
            }

            // Compute mean and inverse standard deviation
            float group_mean = local_sum / group_size;
            float variance = local_sum_sq / group_size - group_mean * group_mean;
            float group_inv_std = 1.0f / sqrtf(variance + eps);

            // Normalize and apply affine transformation
            for (int idx = 0; idx < group_size; ++idx) {
                float gelu_val = out[base + idx];
                float norm = (gelu_val - group_mean) * group_inv_std;
                int ch = idx / hw;
                int global_ch = g * channels_per_group + ch;
                out[base + idx] = norm * gn_weight[global_ch] + gn_bias[global_ch];
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t stride,
    torch::Tensor conv_transpose_weight,
    torch::Tensor conv_transpose_bias,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias,
    int64_t num_groups) {

    // Ensure tensors are contiguous
    x = x.contiguous();
    conv_transpose_weight = conv_transpose_weight.contiguous();
    conv_transpose_bias = conv_transpose_bias.contiguous();
    group_norm_weight = group_norm_weight.contiguous();
    group_norm_bias = group_norm_bias.contiguous();

    // Perform transposed convolution
    auto conv_out = at::conv_transpose2d(x, conv_transpose_weight, conv_transpose_bias, {stride});
    auto output = at::empty_like(conv_out);

    int N = conv_out.size(0);
    int C = conv_out.size(1);
    int H = conv_out.size(2);
    int W = conv_out.size(3);
    int hw = H * W;
    int channels_per_group = C / num_groups;
    int group_size = channels_per_group * hw;

    // Call the CPU implementation
    fused_gelu_group_norm_cpu(
        conv_out.data_ptr<float>(),
        output.data_ptr<float>(),
        group_size,
        hw,
        channels_per_group,
        C,
        num_groups,
        1e-5f,
        group_norm_weight.data_ptr<float>(),
        group_norm_bias.data_ptr<float>()
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused ConvTranspose2d with GELU+GroupNorm with Even Workload Distribution (CPU)");
}