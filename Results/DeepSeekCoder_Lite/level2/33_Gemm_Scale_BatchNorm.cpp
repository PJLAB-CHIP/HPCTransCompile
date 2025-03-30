// cpu_fused_scale_bn.cpp
#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

#define WARP_SIZE 32

// Fused kernel: each thread processes one feature column of the input.
// input: pointer to linear output of shape (batch_size, features)
// scale: scaling factors (per feature)
// running_mean, running_var: running statistics to be updated (per feature)
// gamma, beta: BatchNorm weight and bias (per feature)
// output: result after fused scaling and batchnorm, same shape as input
// eps: epsilon for numerical stability in batch norm
// momentum: momentum for running stats update
void fused_scale_bn_kernel(
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
    float momentum,
    int feature_idx) {

    // Load per-feature parameters into registers
    float s = scale[feature_idx];   // scaling factor
    float g = gamma[feature_idx];   // BN weight
    float b = beta[feature_idx];    // BN bias

    // First pass: compute the mean and variance (after applying scaling) using a thread block
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    // Process batch elements in a grid-stride loop over the batch dimension
    for (int n = 0; n < batch_size; ++n) {
        // Each element is scaled before statistics are computed
        float val = input[n * features + feature_idx] * s;
        sum += val;
        sum_sq += val * val;
    }

    // Use warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Compute the mean and variance
    float mean = sum / batch_size;
    float var = sum_sq / batch_size - mean * mean;

    // Update running statistics (in-place update since one block per feature)
    running_mean[feature_idx] = momentum * running_mean[feature_idx] + (1.0f - momentum) * mean;
    running_var[feature_idx] = momentum * running_var[feature_idx] + (1.0f - momentum) * var;

    // Compute the inverse standard deviation
    float inv_std = 1.0f / sqrt(var + eps);

    // Second pass: perform fused normalization with scaling
    // Formula: output = ((input * s - mean) * inv_std) * g + b
    for (int n = 0; n < batch_size; ++n) {
        float val = input[n * features + feature_idx] * s;
        output[n * features + feature_idx] = ((val - mean) * inv_std) * g + b;
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
    auto device = x.device();
    
    // Ensure tensors are contiguous and on the proper device
    x = x.contiguous();
    gemm_weight = gemm_weight.to(device).contiguous();
    gemm_bias = gemm_bias.to(device).contiguous();
    scale = scale.to(device).contiguous();
    gamma = gamma.to(device).contiguous();
    beta = beta.to(device).contiguous();
    running_mean = running_mean.to(device).contiguous();
    running_var = running_var.to(device).contiguous();

    // Compute linear output (matrix multiply + bias)
    auto linear_output = at::linear(x, gemm_weight, gemm_bias);
    auto output = at::empty_like(linear_output);

    int batch_size = linear_output.size(0);
    int features = linear_output.size(1);

    // Launch one thread per feature
    #pragma omp parallel for
    for (int c = 0; c < features; ++c) {
        fused_scale_bn_kernel(
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
            momentum,
            c
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused scale and BN forward (CPU)");
}