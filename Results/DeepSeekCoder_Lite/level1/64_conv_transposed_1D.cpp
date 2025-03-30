#include <torch/extension.h>
#include <vector>
#include <omp.h>

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

torch::Tensor forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {
    
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
    }

    // Handle optional bias tensor
    if (bias.has_value()) {
        return torch::conv_transpose1d(
            x,
            weight,
            bias.value(),
            stride,
            padding,
            output_padding,
            groups
        );
    } else {
        return torch::conv_transpose1d(
            x,
            weight,
            torch::Tensor(), // Empty tensor for no bias
            stride,
            padding,
            output_padding,
            groups
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed 1D convolution forward (CPU)");
}