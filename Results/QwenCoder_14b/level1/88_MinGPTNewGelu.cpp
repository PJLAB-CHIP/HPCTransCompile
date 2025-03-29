#include <torch/extension.h>
#include <cmath>
#include <omp.h>

void gelu_cpu(const float* x, float* y, int n) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float xi = x[i];
        float x_cubed = xi * xi * xi;
        float inner = xi + coeff * x_cubed;
        inner *= sqrt_2_over_pi;
        float tanh_val = tanhf(inner);
        y[i] = 0.5f * xi * (1.0f + tanh_val);
    }
}

torch::Tensor gelu_forward_cpu(torch::Tensor x) {
    TORCH_CHECK(!x.is_cuda(), "Input tensor must be on CPU");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();

    gelu_cpu(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward_cpu, "GELU forward CPU implementation");
}
