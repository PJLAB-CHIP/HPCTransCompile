#include <torch/extension.h>
#include <omp.h>

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

torch::Tensor leaky_relu_forward_shared(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    #pragma omp parallel for
    for (int idx = 0; idx < n; ++idx) {
        float val = x[idx].item<float>();
        out[idx] = val > 0 ? val : val * negative_slope;
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward_shared, "LeakyReLU forward with shared memory (CPU)");
}