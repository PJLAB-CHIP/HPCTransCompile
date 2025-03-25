#include <torch/extension.h>
#include <cmath>
#include <omp.h>

void swish_cpu(const float* x, float* y, int64_t n) {
    #pragma omp parallel for
    for (int64_t i = 0; i < n; ++i) {
        const float val = x[i];
        const float sigmoid = 1.0f / (1.0f + std::exp(-val));
        y[i] = val * sigmoid;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(!x.is_cuda(), "Input tensor must be on CPU");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    
    swish_cpu(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass (CPU)");
}