#include <torch/extension.h>
#include <cmath>
#include <vector>
#include <omp.h>

// Fused CPU kernel: Performs Group Normalization, Leaky ReLU, and element-wise sum
void fused_gn_lrelu_sum_cpu_kernel(
    float* x,
    int batch_size,
    int num_channels,
    int channels_per_group,
    int num_groups,
    float eps,
    float negative_slope,
    const float* gn_weight,
    const float* gn_bias) {

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < batch_size; ++row) {
        for (int group = 0; group < num_groups; ++group) {
            int group_start = group * channels_per_group;

            // Calculate mean and variance
            float sum = 0.0f;
            float sumsq = 0.0f;
            for (int i = 0; i < channels_per_group; ++i) {
                int idx = row * num_channels + group_start + i;
                float val = x[idx];
                sum += val;
                sumsq += val * val;
            }

            float mean = sum / channels_per_group;
            float var = sumsq / channels_per_group - mean * mean;
            float inv_std = std::sqrt(1.0f / (var + eps));

            // Normalize, apply affine transformation, Leaky ReLU, then element-wise addition
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

    // Linear layer
    x = torch::addmm(fc_bias, x, fc_weight.t());

    int64_t batch_size = x.size(0);
    int64_t num_channels = x.size(1);
    TORCH_CHECK(num_channels % num_groups == 0, "num_groups must divide num_channels");
    int channels_per_group = num_channels / num_groups;

    // Apply fused kernel
    fused_gn_lrelu_sum_cpu_kernel(
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
    m.def("forward", &forward, "Fused Matmul, GroupNorm, LeakyReLU, and Sum kernel with CPU implementation");
}