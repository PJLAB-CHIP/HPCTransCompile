#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <omp.h>

// CPU implementation of the fused_gelu_group_norm_kernel
void fused_gelu_group_norm_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int group_size,       // = channels_per_group * (H*W)
    int hw,               // H * W
    int channels_per_group,
    int C,                // Total channels
    int num_groups,
    float eps,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias) {

    // Each block processes one group. Calculate group indices.
    int group_global = omp_get_thread_num(); // global group index
    int n = group_global / num_groups;  // batch index
    int g = group_global % num_groups;  // group index
    int base = n * C * hw + g * channels_per_group * hw;  // starting offset for this group

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    int hw_elements = hw;
    int group_size_elements = group_size;

    // Scalar processing if vector load is not applicable
    for (int idx = 0; idx < group_size_elements; ++idx) {
        float v = in[base + idx];
        float gelu = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
        out[base + idx] = gelu;
        local_sum += gelu;
        local_sum_sq += gelu * gelu;
    }

    // Reduction for sum and sum of squares
    local_sum /= group_size_elements;
    local_sum_sq = (local_sum_sq / group_size_elements) - local_sum * local_sum + eps;
    local_sum_sq = 1.0f / sqrtf(local_sum_sq);

    // Normalize and apply affine transformation
    for (int idx = 0; idx < group_size_elements; ++idx) {
        float gelu = out[base + idx];
        float norm = (gelu - local_sum) * local_sum_sq;
        int ch = idx / hw;
        int global_ch = g * channels_per_group + ch;
        out[base + idx] = norm * gn_weight[global_ch] + gn_bias[global_ch];
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

    // Ensure tensors are contiguous and on the correct device
    x = x.contiguous();
    conv_transpose_weight = conv_transpose_weight.contiguous();
    conv_transpose_bias = conv_transpose_bias.contiguous();
    group_norm_weight = group_norm_weight.contiguous();
    group_norm_bias = group_norm_bias.contiguous();

    if (!x.is_cuda()) x = x.cuda();
    if (!conv_transpose_weight.is_cuda()) conv_transpose_weight = conv_transpose_weight.cuda();
    if (!conv_transpose_bias.is_cuda()) conv_transpose_bias = conv_transpose_bias.cuda();
    if (!group_norm_weight.is_cuda()) group_norm_weight = group_norm_weight.cuda();
    if (!group_norm_bias.is_cuda()) group_norm_bias = group_norm_bias.cuda();

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

    // Parallelize over groups
    #pragma omp parallel for
    for (int group_global = 0; group_global < N * num_groups; ++group_global) {
        fused_gelu_group_norm_kernel(
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
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused ConvTranspose2d with GELU+GroupNorm with Even Workload Distribution (CPU)");
}