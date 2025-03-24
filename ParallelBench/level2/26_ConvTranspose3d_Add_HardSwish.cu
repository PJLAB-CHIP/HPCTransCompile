#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int VEC_SIZE>
__global__ void fused_add_hardswish_optimized(
    const scalar_t* __restrict__ x_conv,
    const scalar_t* __restrict__ add_input,
    scalar_t* __restrict__ output,
    const size_t num_elements) {
    
    constexpr scalar_t three = static_cast<scalar_t>(3.0);
    constexpr scalar_t sixth = static_cast<scalar_t>(1.0/6.0);
    
    using vec_t = typename std::conditional<
        std::is_same<scalar_t, float>::value,
        float4,
        double2
    >::type;

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t vec_stride = blockDim.x * gridDim.x;
    
    // Vectorized main path
    size_t vec_idx = idx;
    while (vec_idx < num_elements / VEC_SIZE) {
        const vec_t x_vec = __ldg(reinterpret_cast<const vec_t*>(x_conv) + vec_idx);
        const vec_t a_vec = __ldg(reinterpret_cast<const vec_t*>(add_input) + vec_idx);
        vec_t out_vec;

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            const scalar_t temp = reinterpret_cast<const scalar_t*>(&x_vec)[i] 
                                + reinterpret_cast<const scalar_t*>(&a_vec)[i];
            const scalar_t shifted = fmaxf(fminf(temp + three, 6.0f), 0.0f);
            reinterpret_cast<scalar_t*>(&out_vec)[i] = temp * (shifted * sixth) * temp;
        }
        
        reinterpret_cast<vec_t*>(output)[vec_idx] = out_vec;
        vec_idx += vec_stride;
    }

    // Scalar cleanup
    const size_t scalar_idx = idx + (num_elements / VEC_SIZE) * VEC_SIZE;
    if (scalar_idx < num_elements) {
        const scalar_t temp = __ldg(x_conv + scalar_idx) + __ldg(add_input + scalar_idx);
        const scalar_t shifted = fmaxf(fminf(temp + three, 6.0f), 0.0f);
        output[scalar_idx] = temp * (shifted * sixth) * temp;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor add_input,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias) {

    auto x_conv = torch::conv_transpose3d(x, conv_transpose, conv_transpose_bias,
                                        stride, padding, output_padding);

    TORCH_CHECK(x_conv.sizes() == add_input.sizes(), "add_input must match conv output shape");

    auto output = torch::empty_like(x_conv);
    const size_t num_elements = x_conv.numel();

    const int vec_size = (x_conv.scalar_type() == torch::kFloat32) ? 4 : 2;
    const int threads = 128;
    const int num_vec_elements = num_elements / vec_size;
    const int blocks = (num_vec_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x_conv.scalar_type(), "fused_add_hardswish_optimized", ([&] {
        fused_add_hardswish_optimized<scalar_t, (sizeof(scalar_t) == 4) ? 4 : 2><<<blocks, threads>>>(
            x_conv.data_ptr<scalar_t>(),
            add_input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_elements
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused ConvTranspose3D+Add+HardSwish with LDG optimizations");
}
