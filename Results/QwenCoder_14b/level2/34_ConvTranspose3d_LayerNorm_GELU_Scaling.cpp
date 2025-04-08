#include <torch/extension.h>
#include <cmath>
#include <omp.h>

using namespace std;

typedef float4 vec4;

vec4 load_vec4(const float* addr) {
    vec4 result;
    for (int i = 0; i < 4; i++) {
        reinterpret_cast<float*>(&result)[i] = addr[i];
    }
    return result;
}

void store_vec4(float* addr, vec4 val) {
    for (int i = 0; i < 4; i++) {
        addr[i] = reinterpret_cast<float*>(&val)[i];
    }
}

vec4 gelu_vec4(vec4 val) {
    vec4 result;
    for (int i = 0; i < 4; i++) {
        float* v = reinterpret_cast<float*>(&val) + i;
        float* r = reinterpret_cast<float*>(&result) + i;
        float cube = 0.044715f * (*v) * (*v) * (*v);
        float cdf = 0.5f * (1.0f + tanhf(0.797885f * (*v + cube)));
        *r = (*v) * cdf;
    }
    return result;
}

void balanced_gelu_scaling_cpu(
    const float* x,
    float* out,
    const int64_t numel,
    const float scaling_factor
) {
    const int vector_size = 4;
    const int vectors_per_thread = 1024; // Adjust based on your system's capabilities

    #pragma omp parallel for
    for (int vector_idx = 0; vector_idx < numel; vector_idx += vectors_per_thread) {
        for (int i = vector_idx; i < min(vector_idx + vectors_per_thread, numel); i++) {
            const int actual_idx = i * vector_size;

            if (actual_idx < numel) {
                vec4 inputs = load_vec4(x + actual_idx);
                vec4 gelu_result = gelu_vec4(inputs);
                for (int j = 0; j < 4; j++) {
                    reinterpret_cast<float*>(&gelu_result)[j] *= scaling_factor;
                }
                store_vec4(out + actual_idx, gelu_result);
            }
        }
    }
}

void gelu_scaling_forward_cpu(
    at::Tensor& x,
    at::Tensor& out,
    double scaling_factor
) {
    AT_ASSERTM(x.is_contiguous(), "Input tensor must be contiguous");
    AT_ASSERTM(out.is_contiguous(), "Output tensor must be contiguous");

    const int64_t numel = x.numel();
    balanced_gelu_scaling_cpu(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        numel,
        static_cast<float>(scaling_factor)
    );
}

at::Tensor forward(
    at::Tensor x,
    int64_t stride,
    int64_t padding,
    double eps,
    double scaling_factor,
    at::Tensor conv_transpose_weight,
    at::Tensor conv_transpose_bias,
    at::Tensor layer_norm_weight,
    at::Tensor layer_norm_bias
) {
    at::Tensor x_conv = at::conv_transpose3d(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        {stride, stride, stride},
        {padding, padding, padding}
    );

    int64_t out_channels = x_conv.size(1);
    at::Tensor x_norm = at::layer_norm(
        x_conv,
        {out_channels},
        layer_norm_weight,
        layer_norm_bias,
        eps
    );

    at::Tensor x_scaled = at::empty_like(x_norm);
    gelu_scaling_forward_cpu(x_norm, x_scaled, scaling_factor);
    return x_scaled;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced GELU with efficient load balancing");
}
