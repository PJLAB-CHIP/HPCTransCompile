#include <torch/extension.h>
#include <vector>
#include <omp.h>

// Utility function to clamp a value between min_val and max_val
template <typename scalar_t>
inline scalar_t clamp_val(scalar_t value, scalar_t min_val, scalar_t max_val) {
    return value < min_val ? min_val : (value > max_val ? max_val : value);
}

// Combined HardSwish and ReLU operation
// f(x) = max(x * clamp(x+3, 0, 6) / 6, 0)
template <typename scalar_t>
inline scalar_t hard_swish_relu(scalar_t x) {
    scalar_t tmp = clamp_val(x + scalar_t(3), scalar_t(0), scalar_t(6));
    scalar_t hs = x * tmp / scalar_t(6);
    return hs > scalar_t(0) ? hs : scalar_t(0);
}

// CPU forward function
torch::Tensor hardswish_relu_cpu_forward(torch::Tensor input) {
    input = input.contiguous();
    auto num_elements = input.numel();
    auto output = torch::empty_like(input);

    #pragma omp parallel for
    for (int64_t i = 0; i < num_elements; ++i) {
        output[i] = hard_swish_relu<scalar_t>(input[i]);
    }

    return output;
}

// C++ interface: Applies convolution followed by fused HardSwish and ReLU activations
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias) {
    x = torch::conv2d(x, conv_weight, conv_bias);
    x = hardswish_relu_cpu_forward(x);
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Convolution -> Fused HardSwish -> ReLU forward (CPU)");
}