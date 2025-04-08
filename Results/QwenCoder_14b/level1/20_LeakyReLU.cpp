#include <torch/extension.h>
#include <vector>
#include <algorithm>

// Function to apply LeakyReLU activation
void leaky_relu_cpu(const float* x, float* out, float negative_slope, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        out[i] = x[i] > 0 ? x[i] : x[i] * negative_slope;
    }
}

// Wrapper function for PyTorch
torch::Tensor leaky_relu_forward_cpu(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    leaky_relu_cpu(x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward_cpu, "LeakyReLU forward (CPU)");
}