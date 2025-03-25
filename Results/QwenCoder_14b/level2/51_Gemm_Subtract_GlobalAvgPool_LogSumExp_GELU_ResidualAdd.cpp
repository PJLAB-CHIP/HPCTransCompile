#include <torch/extension.h>
#include <cmath>
#include <omp.h>

// GELU approximation function
float gelu_approx(float val) {
    const float kAlpha = 0.044715f;
    const float kBeta  = 0.7978845608f; // sqrt(2/M_PI)
    float inner = kBeta * (val + kAlpha * val * val * val);
    float cdf   = 0.5f * (1.0f + tanhf(inner));
    return val * cdf;
}

// Fused forward function for CPU
torch::Tensor forward_cpu_fused(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& subtract
) {
    TORCH_CHECK(x.device().is_cpu(), "x must be a CPU tensor");
    TORCH_CHECK(weight.device().is_cpu(), "weight must be a CPU tensor");
    TORCH_CHECK(bias.device().is_cpu(), "bias must be a CPU tensor");
    TORCH_CHECK(subtract.device().is_cpu(), "subtract must be a CPU tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (batch_size x in_features)");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D (out_features x in_features)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (out_features)");
    TORCH_CHECK(subtract.dim() == 1, "subtract must be 1D (out_features)");

    int64_t batch_size  = x.size(0);
    int64_t in_features = x.size(1);
    int64_t out_features = weight.size(0);

    TORCH_CHECK(weight.size(1) == in_features, "weight.shape[1] must match x.shape[1]");
    TORCH_CHECK(bias.size(0) == out_features, "bias.shape[0] must match weight.shape[0]");
    TORCH_CHECK(subtract.size(0) == out_features, "subtract.shape[0] must match weight.shape[0]");

    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();
    auto subtract_contig = subtract.contiguous();

    // Precompute weight_sum: sum over rows of weight (weight is out_features x in_features)
    // weight_sum will have shape (in_features,)
    auto weight_sum = torch::sum(weight_contig, 0);

    // Precompute constant = sum(bias - subtract) [a scalar]
    auto constant_tensor = torch::sum(bias_contig - subtract_contig);
    float constant = constant_tensor.item<float>();

    // Allocate output tensor (same shape as x)
    auto out = torch::empty({batch_size, in_features}, x.options());

    #pragma omp parallel for
    for (int row = 0; row < batch_size; ++row) {
        float sum_val = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            float x_val = x_contig.data_ptr<float>()[row * in_features + k];
            float ws = weight_sum.data_ptr<float>()[k];
            sum_val += x_val * ws;
        }

        float pool_val = (sum_val + constant) / static_cast<float>(out_features);
        pool_val = gelu_approx(pool_val);

        for (int k = 0; k < in_features; ++k) {
            out.data_ptr<float>()[row * in_features + k] = x_contig.data_ptr<float>()[row * in_features + k] + pool_val;
        }
    }

    return out;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu_fused, "Fused Forward CPU Function");
}