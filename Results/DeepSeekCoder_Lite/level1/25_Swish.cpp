#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cpu(), "Input tensor must be on CPU");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();

    #pragma omp parallel for
    for (int64_t i = 0; i < n; ++i) {
        const float val = x[i];
        const float sigmoid = 1.0f / (1.0f + expf(-val));
        y[i] = val * sigmoid;
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass (CPU)");
}