#include <torch/extension.h>
#include <omp.h>

// Define a vectorized type for coalesced memory access with alignment for 4 consecutive elements
template <typename scalar_t>
struct alignas(sizeof(scalar_t) * 4) Vec4 {
    scalar_t v[4];
};

// CPU implementation for completeness
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

// Dispatcher function: selects the CUDA or CPU implementation

torch::Tensor even_div_leaky_relu(torch::Tensor x, double divisor, double negative_slope) {
    x = x.contiguous();
    const int64_t size = x.numel();

    if (x.is_cuda()) {
        AT_ERROR("CUDA not supported, only CPU implementation available");
    } else {
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
    }

    return x;
}

// Forward function: applies convolution followed by the evenly distributed division and LeakyReLU operation

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    double divisor
) {
    // Convolution using PyTorch's ATen conv2d
    x = at::conv2d(x, conv_weight, conv_bias);
    // Apply element-wise division and LeakyReLU with negative_slope = 0.01
    x = even_div_leaky_relu(x, divisor, 0.01);
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Convolution, division, and LeakyReLU forward (even load distribution)");
}