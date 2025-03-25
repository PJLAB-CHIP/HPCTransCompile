#include <torch/extension.h>
#include <omp.h>

template <typename scalar_t>
void even_div_leaky_relu_cpu_impl(
    scalar_t* x, scalar_t divisor, scalar_t negative_slope, int64_t size
) {
    #pragma omp parallel for
    for (int64_t i = 0; i < size; i++) {
        scalar_t val = x[i] / divisor;
        x[i] = (val >= static_cast<scalar_t>(0)) ? val : val * negative_slope;
    }
}

torch::Tensor even_div_leaky_relu(torch::Tensor x, double divisor, double negative_slope) {
    x = x.contiguous();
    const int64_t size = x.numel();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "even_div_leaky_relu_cpu", ([&] {
        scalar_t divisor_val = static_cast<scalar_t>(divisor);
        scalar_t negative_slope_val = static_cast<scalar_t>(negative_slope);
        even_div_leaky_relu_cpu_impl<scalar_t>(
            x.data_ptr<scalar_t>(),
            divisor_val,
            negative_slope_val,
            size
        );
    }));

    return x;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    double divisor
) {
    x = at::conv2d(x, conv_weight, conv_bias);
    x = even_div_leaky_relu(x, divisor, 0.01);
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Convolution, division, and LeakyReLU forward (even load distribution)");
}