#include <torch/extension.h>
#include <cmath>
#include <omp.h>

template <typename scalar_t>
float4 tanh_vec4(float4 val) {
    float4 result;
    result.x = tanhf(val.x);
    result.y = tanhf(val.y);
    result.z = tanhf(val.z);
    result.w = tanhf(val.w);
    return result;
}

void tanh_cpu_vectorized(
    const scalar_t* input,
    scalar_t* output,
    const int size) {
    
    const int vec4_size = size / 4;
    
    #pragma omp parallel for
    for (int i = 0; i < vec4_size; ++i) {
        float4 in4 = reinterpret_cast<const float4*>(input)[i];
        reinterpret_cast<float4*>(output)[i] = tanh_vec4(in4);
    }
    
    // Handle remaining elements
    const int remaining_start = vec4_size * 4;
    #pragma omp parallel for
    for (int i = remaining_start; i < size; ++i) {
        output[i] = tanhf(input[i]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    tanh_cpu_vectorized<scalar_t>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        input.numel()
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward vectorized (CPU)");
}