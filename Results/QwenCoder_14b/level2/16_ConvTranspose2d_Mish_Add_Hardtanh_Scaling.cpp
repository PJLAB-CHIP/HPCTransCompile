#include <torch/extension.h>
#include <torch/nn/functional.h>
#include <vector>
#include <cmath>
#include <omp.h>

namespace F = torch::nn::functional;

// Define the mish_hardtanh_scale function for CPU
float mish_hardtanh_scale(float v, float add_value, float scale) {
    float mish = v * tanhf(log1p(exp(v)));
    mish += add_value;
    mish = std::min(std::max(mish, -1.0f), 1.0f);
    return mish * scale;
}

// CPU version of the forward function
torch::Tensor forward_cpu(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    double add_value,
    double scale) {
    
    x = torch::conv_transpose2d(
        x, 
        conv_transpose, 
        conv_transpose_bias, 
        {stride, stride}, 
        {padding, padding}, 
        {output_padding, output_padding});
    
    int size = x.numel();
    int vector_size = size / 4;
    
    #pragma omp parallel for
    for (int i = 0; i < vector_size; i++) {
        int vector_id = i * 4;
        float* ptr = x.data_ptr<float>() + vector_id;
        
        for (int j = 0; j < 4; j++) {
            ptr[j] = mish_hardtanh_scale(ptr[j], static_cast<float>(add_value), static_cast<float>(scale));
        }
    }
    
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "Custom CPU forward function with OpenMP optimization");
}