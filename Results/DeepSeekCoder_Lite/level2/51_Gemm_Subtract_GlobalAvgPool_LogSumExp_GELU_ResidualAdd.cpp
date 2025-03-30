#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

//------------------------------------------------------------------------------
// GELU approximation function
inline float gelu_approx(float val) {
    const float kAlpha = 0.044715f;
    const float kBeta  = 0.7978845608f; // sqrt(2/M_PI)
    float inner = kBeta * (val + kAlpha * val * val * val);
    float cdf   = 0.5f * (1.0f + tanhf(inner));
    return val * cdf;
}

//------------------------------------------------------------------------------
// Fused kernel: Computes the dot product of x[i] and weight_sum with a reduction,
// applies normalization using out_features and constant, then applies GELU,
// and finally performs a residual add with x to produce the final output.
// Each thread processes one row.
void fused_forward_kernel_cpu(
    const float* __restrict__ x,            // Input x: shape (batch_size, in_features)
    const float* __restrict__ weight_sum,     // Precomputed weight_sum: shape (in_features)
    float constant,                           // Precomputed constant: sum(bias - subtract)
    float* __restrict__ out,                  // Output: shape (batch_size, in_features)
    int batch_size,
    int in_features,
    int out_features                        // Needed for normalization
) {
    #pragma omp parallel for
    for (int row = 0; row < batch_size; ++row) {
        float sum_val = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            float x_val = x[row * in_features + k];
            float ws = weight_sum[k];
            sum_val += x_val * ws;
        }

        // Normalize the sum, apply GELU, and broadcast the result
        sum_val = (sum_val + constant) / static_cast<float>(out_features);
        sum_val = gelu_approx(sum_val);

        // Broadcast the result to the corresponding elements in the output tensor
        for (int k = 0; k < in_features; ++k) {
            out[row * in_features + k] = x[row * in_features + k] + sum_val;
        }
    }
}

//------------------------------------------------------------------------------
// Forward function for the fused kernel
// Precomputes the necessary reductions (weight_sum and constant) and launches the fused kernel.

torch::Tensor forward_cpu_fused(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& subtract
) {
    TORCH_CHECK(x.is_cpu(), "x must be a CPU tensor");
    TORCH_CHECK(weight.is_cpu(), "weight must be a CPU tensor");
    TORCH_CHECK(bias.is_cpu(), "bias must be a CPU tensor");
    TORCH_CHECK(subtract.is_cpu(), "subtract must be a CPU tensor");
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

    fused_forward_kernel_cpu(
        x_contig.data_ptr<float>(),
        weight_sum.data_ptr<float>(),
        constant,
        out.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return out;
}

//------------------------------------------------------------------------------
// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu_fused, "Fused Forward CPU Kernel");
}