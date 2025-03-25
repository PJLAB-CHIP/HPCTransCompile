#include <torch/extension.h>
#include <cmath>
#include <omp.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Function to apply swish activation
float swish(float val) {
    return val * (1.0f / (1.0f + std::exp(-val)));
}

// Fused CPU function: applies swish activation + bias addition and then group normalization
torch::Tensor fused_swish_bias_groupnorm_cpu(
    torch::Tensor x,
    const torch::Tensor& bias,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    int num_groups,
    float epsilon) {

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(bias);
    CHECK_CONTIGUOUS(gamma);
    CHECK_CONTIGUOUS(beta);

    int N = x.size(0);
    int C = x.size(1);
    int group_channels = C / num_groups;

    // Parallelize over samples and groups
    #pragma omp parallel for collapse(2)
    for (int sample_idx = 0; sample_idx < N; ++sample_idx) {
        for (int group_idx = 0; group_idx < num_groups; ++group_idx) {
            int base = sample_idx * C + group_idx * group_channels;

            // Step 1: Apply swish activation and bias addition
            for (int i = 0; i < group_channels; ++i) {
                int idx = base + i;
                float val = x[idx].item<float>();
                float sigmoid_val = 1.0f / (1.0f + std::exp(-val));
                float activated = val * sigmoid_val;
                int channel = group_idx * group_channels + i;
                activated += bias[channel].item<float>();
                x[idx] = activated;
            }

            // Step 2: Compute mean and variance
            float sum = 0.0f;
            float sumsq = 0.0f;
            for (int i = 0; i < group_channels; ++i) {
                float v = x[idx].item<float>();
                sum += v;
                sumsq += v * v;
            }
            float mean = sum / group_channels;
            float var = sumsq / group_channels - mean * mean;
            float inv_std = std::sqrt(1.0f / (var + epsilon));

            // Step 3: Normalize and apply group norm parameters
            for (int i = 0; i < group_channels; ++i) {
                int idx = base + i;
                float norm = (x[idx].item<float>() - mean) * inv_std;
                int channel = group_idx * group_channels + i;
                norm = norm * gamma[channel].item<float>() + beta[channel].item<float>();
                x[idx] = norm;
            }
        }
    }

    return x;
}

// Forward function: performs a linear operation followed by the fused activation and normalization.
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor weight_bias,
    torch::Tensor bias,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias,
    int num_groups) {

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(weight);
    CHECK_CONTIGUOUS(weight_bias);
    CHECK_CONTIGUOUS(bias);
    CHECK_CONTIGUOUS(group_norm_weight);
    CHECK_CONTIGUOUS(group_norm_bias);

    // Perform matrix multiplication (linear layer)
    auto x_linear = torch::linear(x, weight, weight_bias);

    // Apply fused activation and normalization
    return fused_swish_bias_groupnorm_cpu(x_linear, bias, group_norm_weight, group_norm_bias, num_groups, 1e-5f);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused Swish activation, bias addition, and group normalization");
}