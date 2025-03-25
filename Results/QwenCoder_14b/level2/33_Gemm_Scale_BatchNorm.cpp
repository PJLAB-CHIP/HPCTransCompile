#include <torch/extension.h>
#include <omp.h>
#include <cmath>

// Fused function: each thread processes one feature (channel).
void fused_scale_bn_cpu(
    const float* input,
    const float* scale,
    float* running_mean,
    float* running_var,
    const float* gamma,
    const float* beta,
    float* output,
    int batch_size,
    int features,
    float eps,
    float momentum) {

    #pragma omp parallel for
    for (int c = 0; c < features; ++c) {
        // Load per-feature parameters into registers
        float s = scale[c];   // scaling factor
        float g = gamma[c];   // BN weight
        float b = beta[c];    // BN bias

        // First pass: compute scaled sum and sum of squares over the batch dimension
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        for (int n = 0; n < batch_size; ++n) {
            // Each element is scaled before statistics are computed
            float val = input[n * features + c] * s;
            sum += val;
            sum_sq += val * val;
        }

        // Compute mean and variance
        float mean = sum / batch_size;
        float var = sum_sq / batch_size - mean * mean;

        // Update running statistics
        running_mean[c] = momentum * running_mean[c] + (1.0f - momentum) * mean;
        running_var[c] = momentum * running_var[c] + (1.0f - momentum) * var;

        // Second pass: perform fused normalization with scaling
        // Formula: output = ((input * s - mean) * inv_std) * g + b
        float inv_std = std::sqrt(1.0f / (var + eps));
        for (int n = 0; n < batch_size; ++n) {
            float val = input[n * features + c] * s;
            output[n * features + c] = ((val - mean) * inv_std) * g + b;
        }
    }
}

// Host forward function
at::Tensor forward(
    at::Tensor x,
    float eps,
    float momentum,
    at::Tensor running_mean,
    at::Tensor running_var,
    at::Tensor gemm_weight,
    at::Tensor gemm_bias,
    at::Tensor scale,
    at::Tensor gamma,   // BatchNorm weight
    at::Tensor beta     // BatchNorm bias
) {
    // Ensure tensors are contiguous and on the CPU
    x = x.contiguous();
    gemm_weight = gemm_weight.contiguous();
    gemm_bias = gemm_bias.contiguous();
    scale = scale.contiguous();
    gamma = gamma.contiguous();
    beta = beta.contiguous();
    running_mean = running_mean.contiguous();
    running_var = running_var.contiguous();

    // Compute linear output (matrix multiply + bias)
    auto linear_output = at::matmul(x, gemm_weight.t()) + gemm_bias;
    auto output = at::empty_like(linear_output);

    int batch_size = linear_output.size(0);
    int features = linear_output.size(1);

    fused_scale_bn_cpu(
        linear_output.data_ptr<float>(),
        scale.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features,
        eps,
        momentum
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused scale and BN forward (CPU)");
}