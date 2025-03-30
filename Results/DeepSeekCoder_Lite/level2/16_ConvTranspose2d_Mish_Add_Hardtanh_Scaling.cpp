#include <torch/extension.h>
#include <torch/nn/functional.h>
#include <vector>
#include <omp.h>

namespace F = torch::nn::functional;

__forceinline__ float4 load_float4(const float* ptr) {
    float4 v;
    v.x = ptr[0];
    v.y = ptr[1];
    v.z = ptr[2];
    v.w = ptr[3];
    return v;
}

__forceinline__ void store_float4(float* ptr, float4 v) {
    ptr[0] = v.x;
    ptr[1] = v.y;
    ptr[2] = v.z;
    ptr[3] = v.w;
}

__forceinline__ float4 mish_hardtanh_scale(float4 v, float add_value, float scale) {
    float4 result;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float* val = ((float*)&v) + i;
        float mish = *val * tanhf(log1pf(expf(*val)));
        mish += add_value;
        mish = fminf(fmaxf(mish, -1.0f), 1.0f);
        ((float*)&result)[i] = mish * scale;
    }
    return result;
}

torch::Tensor forward(
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
    
    // Ensure memory alignment
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(x.data_ptr<float>()) % 16 == 0,
        "Input tensor must be 16-byte aligned"
    );
    
    #pragma omp parallel for
    for (int vector_id = 0; vector_id < vector_size; vector_id++) {
        const float* ptr = x.data_ptr<float>() + vector_id * 4;
        float4 v = load_float4(ptr);
        v = mish_hardtanh_scale(v, static_cast<float>(add_value), static_cast<float>(scale));
        store_float4(ptr, v);
    }
    
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Custom CPU forward function with warp optimization");
}