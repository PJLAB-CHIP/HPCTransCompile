#include <torch/extension.h>
#include <cmath>
#include <vector>
#include <omp.h>

// GELU activation function (approximation used in PyTorch)
float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coef = 0.044715f;
    float cdf = 0.5f * (1.0f + tanhf(sqrt_2_over_pi * x * (1.0f + coef * x * x)));
    return x * cdf;
}

// Fused function: Performs linear transformation, applies GELU activation, and softmax normalization
torch::Tensor fused_cpu_function(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
) {
    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = weight.size(0);

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({batch_size, out_features}, options);

    #pragma omp parallel for
    for (int row = 0; row < batch_size; ++row) {
        std::vector<float> act(out_features, 0.0f);
        std::vector<float> exp_vals(out_features, 0.0f);

        // 1. Compute the dot product for the linear transformation for each valid output feature
        for (int tid = 0; tid < out_features; ++tid) {
            float sum = 0.0f;
            for (int k = 0; k < in_features; k++) {
                sum += x[row * in_features + k] * weight[tid * in_features + k];
            }
            sum += bias[tid];
            act[tid] = gelu(sum);
            exp_vals[tid] = expf(act[tid]);
        }

        // 2. Reduction to compute the maximum activated value across the outputs (for softmax numerical stability)
        float row_max = *std::max_element(act.begin(), act.end());

        // 3. Compute the exponentials; invalid threads (tid >= out_features) produce 0
        for (int tid = 0; tid < out_features; ++tid) {
            exp_vals[tid] = expf(act[tid] - row_max);
        }

        // 4. Reduction to compute the sum of exponentials
        float sum_exp = std::accumulate(exp_vals.begin(), exp_vals.end(), 0.0f);

        // 5. Write the normalized softmax result for valid output features
        for (int tid = 0; tid < out_features; ++tid) {
            output[row * out_features + tid] = exp_vals[tid] / sum_exp;
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_cpu_function, "Fused Linear + GELU + Softmax forward on CPU");
}