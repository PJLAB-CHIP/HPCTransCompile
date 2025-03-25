#include <torch/extension.h>
#include <cmath>
#include <omp.h>

// Explicit specializations of gelu_function for float
template <typename scalar_t>
inline scalar_t gelu_function(scalar_t x) {
    return x * 0.5f * (1.0f + std::erff(x / 1.4142135623730951f));
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(!x.is_cuda(), "Input tensor must be a CPU tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float,
               "Only float32 is supported");

    auto output = torch::empty_like(x);
    const size_t numel = x.numel();

    #pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        output[i] = gelu_function(x[i]);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CPU)");
}