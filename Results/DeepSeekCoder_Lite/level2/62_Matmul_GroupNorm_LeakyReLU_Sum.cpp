#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// Fused kernel: Performs Group Normalization, Leaky ReLU, and element-wise sum
// utilizing warp-level primitives for small reductions to minimize shared memory usage.
void fused_gn_lrelu_sum_warp_kernel(
    float* __restrict__ x,
    int batch_size,
    int num_channels,
    int channels_per_group,
    int num_groups,
    float eps,
    float negative_slope,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias) {

    #pragma omp parallel for
    for (int row = 0; row < batch_size; ++row) {
        for (int group = 0; group < num_groups; ++group) {
            int group_start = group * channels_per_group;
            float sum = 0.0f;
            float sumsq = 0.0f;

            #pragma omp simd reduction(+:sum) reduction(+:sumsq)
            for (int i = 0; i < channels_per_group; ++i) {
                int idx = row * num_channels + group_start + i;
                float val = x[idx];
                sum += val;
                sumsq += val * val;
            }

            // Use warp-level primitives for reduction
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __builtin_ia32_addss(sum, _mm_shuffle_ps(_mm_load_ss(&sum), _mm_load_ss(&sum), offset));
                sumsq += __builtin_ia32_addss(sumsq, _mm_shuffle_ps(_mm_load_ss(&sumsq), _mm_load_ss(&sumsq), offset));
            }

            float mean = sum / channels_per_group;
            float var = sumsq / channels_per_group - mean * mean;
            float inv_std = 1.0f / sqrtf(var + eps);

            // Normalize, apply affine transformation, Leaky ReLU, then element-wise addition
            #pragma omp simd
            for (int i = 0; i < channels_per_group; ++i) {
                int idx = row * num_channels + group_start + i;
                float val = x[idx];
                float norm = (val - mean) * inv_std;
                norm = norm * gn_weight[group_start + i] + gn_bias[group_start + i];
                norm = (norm < 0.0f) ? negative_slope * norm : norm; // Leaky ReLU
                // Element-wise sum: doubling the value
                x[idx] = norm + norm;
            }
        }
    }
}

// Forward function integrates linear transformation and fused group norm operations
// Linear layer: x = fc_bias + x * fc_weight^T
// Followed by fused kernel that applies GroupNorm, LeakyReLU and sum with warp-level reduction
torch::Tensor forward(
    torch::Tensor x,
    double eps,
    double negative_slope,
    torch::Tensor fc_weight,
    torch::Tensor fc_bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    int64_t num_groups) {

    TORCH_CHECK(x.is_cpu(), "x must be a CPU tensor");
    TORCH_CHECK(fc_weight.is_cpu(), "fc_weight must be a CPU tensor");
    TORCH_CHECK(fc_bias.is_cpu(), "fc_bias must be a CPU tensor");
    TORCH_CHECK(gn_weight.is_cpu(), "gn_weight must be a CPU tensor");
    TORCH_CHECK(gn_bias.is_cpu(), "gn_bias must be a CPU tensor");

    // Linear layer
    x = torch::addmm(fc_bias, x, fc_weight.t());

    int64_t batch_size = x.size(0);
    int64_t num_channels = x.size(1);
    TORCH_CHECK(num_channels % num_groups == 0, "num_groups must divide num_channels");
    int channels_per_group = num_channels / num_groups;

    fused_gn_lrelu_sum_warp_kernel(
        x.data_ptr<float>(),
        batch_size,
        num_channels,
        channels_per_group,
        num_groups,
        static_cast<float>(eps),
        static_cast<float>(negative_slope),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>()
    );

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused Matmul, GroupNorm, LeakyReLU, and Sum kernel with warp-level reduction");
}