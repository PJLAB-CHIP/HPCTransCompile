#include <torch/extension.h>
#include <omp.h>
#include <cmath>

template <typename scalar_t>
void fused_add_hardswish_optimized_cpu(
    const scalar_t* x_conv,
    const scalar_t* add_input,
    scalar_t* output,
    const size_t num_elements) {
    
    constexpr scalar_t three = static_cast<scalar_t>(3.0);
    constexpr scalar_t sixth = static_cast<scalar_t>(1.0/6.0);

    #pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++i) {
        const scalar_t temp = x_conv[i] + add_input[i];
        const scalar_t shifted = std::max(std::min(temp + three, 6.0), 0.0);
        output[i] = temp * (shifted * sixth) * temp;
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

    AT_DISPATCH_FLOATING_TYPES(x_conv.scalar_type(), "fused_add_hardswish_optimized_cpu", ([&] {
        fused_add_hardswish_optimized_cpu<scalar_t>(
            x_conv.data_ptr<scalar_t>(),
            add_input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_elements
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused ConvTranspose3D+Add+HardSwish on CPU");
}