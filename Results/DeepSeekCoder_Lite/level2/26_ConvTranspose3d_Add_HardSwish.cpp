#include <torch/extension.h>
#include <omp.h>
#include <algorithm>
#include <cmath>

template <typename scalar_t, int VEC_SIZE>
void fused_add_hardswish_optimized_cpu(
    const scalar_t* x_conv,
    const scalar_t* add_input,
    scalar_t* output,
    const size_t num_elements,
    const int vec_size,
    const int threads) {
    
    constexpr scalar_t three = static_cast<scalar_t>(3.0);
    constexpr scalar_t sixth = static_cast<scalar_t>(1.0/6.0);

    const int num_vec_elements = num_elements / vec_size;
    const int num_blocks = (num_vec_elements + threads - 1) / threads;

    #pragma omp parallel for
    for (int block = 0; block < num_blocks; ++block) {
        const int idx = block * threads;
        const int vec_idx = idx / vec_size;
        const int thread_idx = idx % vec_size;

        if (vec_idx < num_vec_elements) {
            scalar_t temp = x_conv[vec_idx * vec_size + thread_idx] + add_input[vec_idx * vec_size + thread_idx];
            temp = std::fmaxf(std::fminf(temp + three, 6.0f), 0.0f);
            output[vec_idx * vec_size + thread_idx] = temp * (temp * sixth) * temp;
        }

        for (int i = thread_idx + vec_size; i < vec_size; i += threads) {
            if (vec_idx + (i / vec_size) * num_vec_elements < num_vec_elements) {
                scalar_t temp = x_conv[(vec_idx + (i / vec_size) * num_vec_elements) * vec_size + i] + add_input[(vec_idx + (i / vec_size) * num_vec_elements) * vec_size + i];
                temp = std::fmaxf(std::fminf(temp + three, 6.0f), 0.0f);
                output[(vec_idx + (i / vec_size) * num_vec_elements) * vec_size + i] = temp * (temp * sixth) * temp;
            }
        }
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

    AT_DISPATCH_FLOATING_TYPES(x_conv.scalar_type(), "fused_add_hardswish_optimized_cpu", ([&] {
        fused_add_hardswish_optimized_cpu<scalar_t, (sizeof(scalar_t) == 4) ? 4 : 2>(
            x_conv.data_ptr<scalar_t>(),
            add_input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_elements,
            vec_size,
            threads
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused ConvTranspose3D+Add+HardSwish with CPU optimizations");
}