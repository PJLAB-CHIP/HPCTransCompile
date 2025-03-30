#include <torch/extension.h>
#include <vector>
#include <omp.h>

// Utility function to clamp a value between min_val and max_val
template <typename scalar_t>
__forceinline__ scalar_t clamp_val(scalar_t value, scalar_t min_val, scalar_t max_val) {
    return value < min_val ? min_val : (value > max_val ? max_val : value);
}

// Combined HardSwish and ReLU operation
// f(x) = max(x * clamp(x+3, 0, 6) / 6, 0)
template <typename scalar_t>
__forceinline__ scalar_t hard_swish_relu(scalar_t x) {
    scalar_t tmp = clamp_val(x + scalar_t(3), scalar_t(0), scalar_t(6));
    scalar_t hs = x * tmp / scalar_t(6);
    return hs > scalar_t(0) ? hs : scalar_t(0);
}

// Using float4 for vectorized memory accesses
typedef float4 vec_t;

// Fused CPU kernel that applies HardSwish and then ReLU in one pass using vectorized loads/stores
template <typename scalar_t>
void fused_hardswish_relu_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t num_elements) {

    const int vec_elements = num_elements / 4;
    const vec_t* input_vec = reinterpret_cast<const vec_t*>(input);
    vec_t* output_vec = reinterpret_cast<vec_t*>(output);

    #pragma omp parallel for
    for (int i = 0; i < vec_elements; ++i) {
        vec_t in = input_vec[i];
        vec_t out;
        
        // Process each component in the float4
        float* in_f = reinterpret_cast<float*>(&in);
        float* out_f = reinterpret_cast<float*>(&out);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            out_f[j] = hard_swish_relu<float>(in_f[j]);
        }
        
        output_vec[i] = out;
    }

    // Process any remaining elements
    int remaining_start = vec_elements * 4;
    for (int i = remaining_start; i < num_elements; ++i) {
        output[i] = hard_swish_relu<scalar_t>(input[i]);
    }
}

// CPU forward function
torch::Tensor hardswish_relu_cuda_forward(torch::Tensor input) {
    input = input.contiguous();
    auto num_elements = input.numel();
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int vec_elements = num_elements / 4;
    const int blocks = (vec_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "hardswish_relu_cuda_forward", ([&] {
        fused_hardswish_relu_kernel<scalar_t>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_elements);
    }));

    return output;
}

// C++ interface: Applies convolution followed by fused HardSwish and ReLU activations
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias) {
    x = torch::conv2d(x, conv_weight, conv_bias);
    x = hardswish_relu_cuda_forward(x);
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Convolution -> Fused HardSwish -> ReLU forward (CPU)");
}