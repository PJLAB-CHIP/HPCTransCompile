#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int num_elements = x.numel();

    #pragma omp parallel for
    for (int idx = 0; idx < num_elements; ++idx) {
        float val = x[idx].item<float>();
        out[idx] = val / (1.0f + std::fabs(val));
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Softsign activation (CPU)");
}