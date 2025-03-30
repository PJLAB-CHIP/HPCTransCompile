#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

torch::Tensor elu_cpu(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    
    int n = x.numel();
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float val = x[i].item<float>();
        float result = (val > 0) ? val : alpha * (expf(val) - 1);
        out[i] = result;
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cpu, "ELU activation (CPU)");
}