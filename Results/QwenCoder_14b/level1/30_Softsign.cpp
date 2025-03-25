#include <torch/extension.h>
#include <cmath>
#include <omp.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor forward(torch::Tensor x) {
    CHECK_CONTIGUOUS(x);

    auto out = torch::empty_like(x);
    int num_elements = x.numel();

    #pragma omp parallel for
    for (int i = 0; i < num_elements; ++i) {
        float val = x[i].item<float>();
        out[i] = val / (1.0f + std::fabs(val));
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Softsign activation (CPU)");
}