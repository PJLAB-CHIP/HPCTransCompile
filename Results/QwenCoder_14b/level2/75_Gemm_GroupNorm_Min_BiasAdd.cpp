#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cmath>

namespace F = torch::nn::functional;

// Function to compute the minimum value in a vector
float compute_min(const std::vector<float>& data) {
    float min_val = FLT_MAX;
    for (const auto& val : data) {
        min_val = std::min(min_val, val);
    }
    return min_val;
}

// Function to perform group normalization
void group_normalization(
    const torch::Tensor& input,
    torch::Tensor& output,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    const int num_groups,
    const int channels_per_group) {

    int batch_size = input.size(0);
    int channels = input.size(1);

    #pragma omp parallel for
    for (int bid = 0; bid < batch_size; ++bid) {
        const float* row_start = input.data_ptr<float>() + bid * channels;
        std::vector<float> means(num_groups, 0.0f);
        std::vector<float> vars(num_groups, 0.0f);

        // Compute group statistics: mean and variance (std. deviation) for each group
        #pragma omp parallel for
        for (int g = 0; g < num_groups; ++g) {
            float sum = 0.0f, sum_sq = 0.0f;
            int start = g * channels_per_group;
            int end = start + channels_per_group;
            for (int c = start; c < end; ++c) {
                float v = row_start[c];
                sum += v;
                sum_sq += v * v;
            }
            means[g] = sum / channels_per_group;
            float variance = sum_sq / channels_per_group - means[g] * means[g];
            vars[g] = std::sqrt(variance + 1e-5f);
        }

        // Normalize, transform, and compute minimum
        float thread_min = FLT_MAX;
        #pragma omp parallel for
        for (int c = 0; c < channels; ++c) {
            int group = c / channels_per_group;
            float norm = (row_start[c] - means[group]) / vars[group];
            float transformed = gamma.data_ptr<float>()[c] * norm + beta.data_ptr<float>()[c];
            thread_min = std::min(thread_min, transformed);
        }

        // Store the minimum value with bias added
        output.data_ptr<float>()[bid] = thread_min + bias.data_ptr<float>()[bid];
    }
}

// Forward function: Fuses GEMM, Group Normalization and min reduction
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor gemm_weight,
    torch::Tensor gemm_bias,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias,
    int64_t num_groups,
    torch::Tensor bias) {

    // GEMM: perform linear transformation
    x = F::linear(x, gemm_weight, gemm_bias);

    const int batch_size = x.size(0);
    const int channels = x.size(1);
    const int channels_per_group = channels / num_groups;

    auto output = torch::empty({batch_size}, x.options());

    // Perform group normalization and min reduction
    group_normalization(x, output, group_norm_weight, group_norm_bias, num_groups, channels_per_group);

    return output.unsqueeze(1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused GroupNorm and Min Reduction with GEMM");
}