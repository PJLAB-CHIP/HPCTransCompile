#include <torch/extension.h>
#include <cmath>
#include <omp.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor elu_cpu(torch::Tensor x, float alpha) {
    CHECK_CONTIGUOUS(x);
    auto out = torch::empty_like(x);
    
    int n = x.numel();
    int n4 = n / 4;  // Number of float4 elements
    
    #pragma omp parallel for
    for (int i = 0; i < n4; ++i) {
        float4 val = reinterpret_cast<float4*>(x.data_ptr<float>())[i];
        float4 result;
        
        result.x = (val.x > 0) ? val.x : alpha * (expf(val.x) - 1);
        result.y = (val.y > 0) ? val.y : alpha * (expf(val.y) - 1);
        result.z = (val.z > 0) ? val.z : alpha * (expf(val.z) - 1);
        result.w = (val.w > 0) ? val.w : alpha * (expf(val.w) - 1);
        
        reinterpret_cast<float4*>(out.data_ptr<float>())[i] = result;
    }
    
    // Handle any remaining elements that are not a multiple of 4
    #pragma omp parallel for
    for (int i = n4 * 4; i < n; ++i) {
        float val = x[i].item<float>();
        float result = (val > 0) ? val : alpha * (expf(val) - 1);
        out[i] = result;
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cpu, "ELU activation (CPU)");
}