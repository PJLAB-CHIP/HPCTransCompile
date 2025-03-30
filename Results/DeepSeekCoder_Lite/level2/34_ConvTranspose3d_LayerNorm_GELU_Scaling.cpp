#include <torch/extension.h>
#include <omp.h>
#include <cmath>

// Vector type for 4-wide operations
typedef float4 vec4;

__forceinline__ vec4 load_vec4(const float* addr) {
    return *reinterpret_cast<const vec4*>(addr);
}

__forceinline__ void store_vec4(float* addr, vec4 val) {
    *reinterpret_cast<vec4*>(addr) = val;
}

__forceinline__ vec4 gelu_vec4(vec4 val) {
    vec4 result;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float* v = reinterpret_cast<float*>(&val) + i;
        float* r = reinterpret_cast<float*>(&result) + i;
        float cube = 0.044715f * (*v) * (*v) * (*v);
        float cdf = 0.5f * (1.0f + tanhf(0.797885f * (*v + cube)));
        *r = (*v) * cdf;
    }
    return result;
}

void gelu_scaling_forward_cpu(
    const float* x,
    float* out,
    const int64_t numel,
    const float scaling_factor,
    const int vector_size = 4
) {
    #pragma omp parallel for
    for (int vector_idx = 0; vector_idx < numel; vector_idx += vector_size) {
        int actual_idx = vector_idx * vector_size;
        if (actual_idx < numel) {
            vec4 inputs = load_vec4(x + actual_idx);
            vec4 gelu_result = gelu_vec4(inputs);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                reinterpret_cast<float*>(&gelu_result)[i] *= scaling_factor;
            }
            store_vec4(out + actual_idx, gelu_result);
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    double eps,
    double scaling_factor,
    torch::Tensor conv_transpose_weight,
    torch::Tensor conv_transpose_bias,
    torch::Tensor layer_norm_weight,
    torch::Tensor layer_norm_bias
) {
    torch::Tensor x_conv = at::conv_transpose3d(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        {stride, stride, stride},
        {padding, padding, padding}
    );

    int64_t out_channels = x_conv.size(1);
    torch::Tensor x_norm = at::layer_norm(
        x_conv,
        {out_channels},
        layer_norm_weight,
        layer_norm_bias,
        eps
    );

    torch::Tensor x_scaled = torch::empty_like(x_norm);
    gelu_scaling_forward_cpu(
        x_norm.data_ptr<float>(),
        x_scaled.data_ptr<float>(),
        x_norm.numel(),
        static_cast<float>(scaling_factor)
    );
    return x_scaled;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced GELU with efficient load balancing");
}