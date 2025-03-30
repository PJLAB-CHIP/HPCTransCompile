#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

template <typename scalar_t>
__forceinline__ float tanh_scalar(float val) {
    return tanhf(val);
}

template <typename scalar_t>
void tanh_kernel_vectorized(
    const scalar_t* input,
    scalar_t* output,
    const int size) {
    
    const int vec4_size = size / 4;
    
    // Process 4 elements at a time using float4
    for (int i = 0; i < vec4_size; ++i) {
        float4 in4;
        in4.x = input[i * 4 + 0];
        in4.y = input[i * 4 + 1];
        in4.z = input[i * 4 + 2];
        in4.w = input[i * 4 + 3];
        
        float4 out4;
        out4.x = tanh_scalar(in4.x);
        out4.y = tanh_scalar(in4.y);
        out4.z = tanh_scalar(in4.z);
        out4.w = tanh_scalar(in4.w);
        
        output[i * 4 + 0] = out4.x;
        output[i * 4 + 1] = out4.y;
        output[i * 4 + 2] = out4.z;
        output[i * 4 + 3] = out4.w;
    }
    
    // Handle remaining elements
    const int remaining_start = vec4_size * 4;
    for (int i = remaining_start; i < size; ++i) {
        output[i] = tanh_scalar(input[i]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int size = input.numel();
    const int threads = 256;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_vectorized", ([&] {
        #pragma omp parallel for
        for (int i = 0; i < size; i += threads) {
            int tid = omp_get_thread_num();
            int start = i + tid;
            int end = std::min(start + threads, size);
            tanh_kernel_vectorized<scalar_t>(
                input.data_ptr<scalar_t>() + start,
                output.data_ptr<scalar_t>() + start,
                end - start
            );
        }
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward vectorized (CPU)");
}