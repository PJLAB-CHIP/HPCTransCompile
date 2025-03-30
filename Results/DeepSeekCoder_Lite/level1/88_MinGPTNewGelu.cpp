#include <torch/extension.h>
#include <math.h>
#include <omp.h>

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.device().is_cpu(), "Input tensor must be on CPU");

    auto y = torch::empty_like(x);
    int n = x.numel();

    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    #pragma omp parallel for
    for (int idx = 0; idx < n; ++idx) {
        float xi = x[idx].item<float>();
        float x_cubed = xi * xi * xi;
        float inner = xi + coeff * x_cubed;
        inner *= sqrt_2_over_pi;
        float tanh_val = tanhf(inner);
        y[idx] = 0.5f * xi * (1.0f + tanh_val);
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CPU implementation with OpenMP");
}