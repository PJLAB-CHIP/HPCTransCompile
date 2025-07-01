#include <torch/extension.h>
#include <torch/nn/functional.h>
#include <vector>

namespace F = torch::nn::functional;

__device__ __forceinline__ float4 load_float4(const float* ptr) {
    float4 v;
    v.x = ptr[0];
    v.y = ptr[1];
    v.z = ptr[2];
    v.w = ptr[3];
    return v;
}

__device__ __forceinline__ void store_float4(float* ptr, float4 v) {
    ptr[0] = v.x;
    ptr[1] = v.y;
    ptr[2] = v.z;
    ptr[3] = v.w;
}

__device__ __forceinline__ float4 mish_hardtanh_scale(float4 v, float add_value, float scale) {
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

__global__ void mish_hardtanh_scaling_kernel_vectorized(
    float* __restrict__ x,
    const int size,
    const float add_value,
    const float scale) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int vector_id = tid * 4;
    
    if (vector_id < size) {
        float4 v = load_float4(x + vector_id);
        v = mish_hardtanh_scale(v, add_value, scale);
        store_float4(x + vector_id, v);
    }
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
    int threads = 256;
    int blocks = (vector_size + threads - 1) / threads;
    
    // Ensure memory alignment
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(x.data_ptr<float>()) % 16 == 0,
        "Input tensor must be 16-byte aligned"
    );
    
    mish_hardtanh_scaling_kernel_vectorized<<<blocks, threads>>>(
        x.data_ptr<float>(),
        size,
        static_cast<float>(add_value),
        static_cast<float>(scale)
    );
    
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Custom CUDA forward function with warp optimization");
}