#include <torch/extension.h>
#include <cmath>
#include <omp.h>

template <typename scalar_t>
void fused_activation_cpu(
    scalar_t* x,
    scalar_t scaling_factor,
    scalar_t hardtanh_min,
    scalar_t hardtanh_max,
    int64_t numel) {
#pragma omp parallel for
    for (int64_t i = 0; i < numel; ++i) {
        scalar_t val = x[i];
        // Scaling
        val = val * scaling_factor;
        // Hardtanh
        val = std::min(std::max(val, hardtanh_min), hardtanh_max);
        // GELU approximation
        const scalar_t c = static_cast<scalar_t>(0.044715);
        const scalar_t sqrt_2_over_pi = static_cast<scalar_t>(0.7978845608028654); // sqrt(2.0 / pi)
        scalar_t x_cube = val * val * val;
        scalar_t tanh_arg = sqrt_2_over_pi * (val + c * x_cube);
        scalar_t tanh_res = std::tanh(tanh_arg);
        val = static_cast<scalar_t>(0.5) * val * (static_cast<scalar_t>(1.0) + tanh_res);
        x[i] = val;
    }
}

void fused_activation(
    torch::Tensor& x,
    double scaling_factor,
    double hardtanh_min,
    double hardtanh_max) {
    const auto numel = x.numel();
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_activation", ([&] {
        fused_activation_cpu<scalar_t>(
            x.data_ptr<scalar_t>(),
            static_cast<scalar_t>(scaling_factor),
            static_cast<scalar_t>(hardtanh_min),
            static_cast<scalar_t>(hardtanh_max),
            numel);
    }));
}

torch::Tensor module_fn_forward(
    torch::Tensor x,
    double scaling_factor,
    double hardtanh_min,
    double hardtanh_max,
    torch::Tensor weight,
    torch::Tensor bias) {

    // Ensure inputs are contiguous and on CPU
    x = x.contiguous().cpu();
    weight = weight.contiguous().cpu();
    bias = bias.contiguous().cpu();

    // Linear transformation: x = x @ weight.T + bias
    auto xw = torch::matmul(x, weight.t()) + bias;

    // Apply fused activation functions
    fused_activation(xw, scaling_factor, hardtanh_min, hardtanh_max);

    return xw;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Module function forward (CPU)");
}