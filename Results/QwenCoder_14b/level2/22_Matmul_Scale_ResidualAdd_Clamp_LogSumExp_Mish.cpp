#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

// Define block size for kernels
constexpr int BLOCK_SIZE = 256;

// Scalar function to clamp and scale
void clamp_and_scale_scalar(const float* in, float* out, int num_elements, float factor, float min_val, float max_val) {
    #pragma omp parallel for
    for (int idx = 0; idx < num_elements; ++idx) {
        float v = in[idx];
        v = v * (2.0f * factor);
        v = std::fmin(std::fmax(v, min_val), max_val);
        out[idx] = v;
    }
}

// Vectorized function processing 4 floats at a time
void clamp_and_scale_vectorized(const float* in, float* out, int num_elements4, float factor, float min_val, float max_val) {
    #pragma omp parallel for
    for (int idx = 0; idx < num_elements4; ++idx) {
        float v0 = in[idx * 4 + 0];
        float v1 = in[idx * 4 + 1];
        float v2 = in[idx * 4 + 2];
        float v3 = in[idx * 4 + 3];
        float s = 2.0f * factor;
        v0 = std::fmin(std::fmax(v0 * s, min_val), max_val);
        v1 = std::fmin(std::fmax(v1 * s, min_val), max_val);
        v2 = std::fmin(std::fmax(v2 * s, min_val), max_val);
        v3 = std::fmin(std::fmax(v3 * s, min_val), max_val);
        out[idx * 4 + 0] = v0;
        out[idx * 4 + 1] = v1;
        out[idx * 4 + 2] = v2;
        out[idx * 4 + 3] = v3;
    }
}

// Function to perform LogSumExp across rows and apply Mish activation
void logsumexp_mish_function(const float* input, float* output, int rows, int cols) {
    #pragma omp parallel for
    for (int row = 0; row < rows; ++row) {
        // Find maximum value in the row
        float max_val = -INFINITY;
        for (int i = 0; i < cols; ++i) {
            float val = input[row * cols + i];
            max_val = std::fmax(max_val, val);
        }

        // Compute the sum of exp(value - max) for numerical stability
        float sum = 0.0f;
        for (int i = 0; i < cols; ++i) {
            float v = input[row * cols + i];
            sum += std::exp(v - max_val);
        }

        float lse = std::log(sum) + max_val;
        
        // Apply Mish activation: mish(x) = x * tanh(softplus(x)) => final: x * (x * tanh(softplus(x)))
        float softplus = std::log1p(std::exp(lse));
        float mish = lse * std::tanh(softplus);
        output[row] = lse * mish;
    }
}

// Forward function that implements the complete fused operation
torch::Tensor module_fn_forward(
    torch::Tensor x,
    float scale_factor,
    float clamp_min,
    float clamp_max,
    torch::Tensor weight,
    torch::Tensor bias
) {
    // Ensure inputs are contiguous for aligned memory accesses
    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    // 1. Matrix multiplication and bias addition
    auto out = torch::mm(x, weight.transpose(0, 1));
    out.add_(bias);

    // 2. Fuse scaling, residual addition, and clamping using a custom function
    int num_elements = out.numel();
    // Check for 128-bit alignment and divisibility by 4 for vectorized operations
    bool use_vectorized = (num_elements % 4 == 0) && (((uintptr_t)out.data_ptr<float>()) % 16 == 0);

    if (use_vectorized) {
        int num_elements4 = num_elements / 4;
        clamp_and_scale_vectorized(out.data_ptr<float>(), out.data_ptr<float>(), num_elements4, scale_factor, clamp_min, clamp_max);
    } else {
        clamp_and_scale_scalar(out.data_ptr<float>(), out.data_ptr<float>(), num_elements, scale_factor, clamp_min, clamp_max);
    }
    
    // 3. Apply LogSumExp and Mish activation along rows using a reduction function
    auto output = torch::empty({out.size(0), 1}, out.options());
    logsumexp_mish_function(out.data_ptr<float>(), output.data_ptr<float>(), out.size(0), out.size(1));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Forward pass for module_fn (CPU)");
}