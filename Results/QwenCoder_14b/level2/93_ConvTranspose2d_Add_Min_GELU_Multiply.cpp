#include <torch/extension.h>
#include <cmath>
#include <omp.h>

// Helper functions
float apply_ops(float val, float add_value, float multiply_value) {
    val = val + add_value;
    val = std::fmin(val, 0.0f);
    float t = tanhf(0.79788456f * (val + 0.044715f * val * val * val));
    return (val * 0.5f * (1.0f + t)) * multiply_value;
}

double apply_ops(double val, double add_value, double multiply_value) {
    val = val + add_value;
    val = (val < 0.0) ? val : 0.0;
    double t = tanh(0.79788456 * (val + 0.044715 * val * val * val));
    return (val * 0.5 * (1.0 + t)) * multiply_value;
}

// CPU implementation of the elementwise operations
void elementwise_cpu(float* x, int64_t size, float add_value, float multiply_value) {
    #pragma omp parallel for
    for (int64_t i = 0; i < size; ++i) {
        x[i] = apply_ops(x[i], add_value, multiply_value);
    }
}

void elementwise_cpu(double* x, int64_t size, double add_value, double multiply_value) {
    #pragma omp parallel for
    for (int64_t i = 0; i < size; ++i) {
        x[i] = apply_ops(x[i], add_value, multiply_value);
    }
}

// Main function: applies conv_transpose2d then elementwise operations
torch::Tensor module_fn(
    torch::Tensor x,
    int64_t stride,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    double add_value,
    double multiply_value
) {
    // Ensure tensor is contiguous
    x = x.contiguous();
    auto numel = x.numel();

    // Apply transposed convolution
    x = at::conv_transpose2d(x, conv_transpose, conv_transpose_bias, {stride});

    // Apply elementwise operations using our CPU implementation
    if (x.scalar_type() == at::ScalarType::Float) {
        elementwise_cpu(x.data_ptr<float>(), numel, static_cast<float>(add_value), static_cast<float>(multiply_value));
    } else if (x.scalar_type() == at::ScalarType::Double) {
        elementwise_cpu(x.data_ptr<double>(), numel, add_value, multiply_value);
    }

    return x;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Module function forward");
}