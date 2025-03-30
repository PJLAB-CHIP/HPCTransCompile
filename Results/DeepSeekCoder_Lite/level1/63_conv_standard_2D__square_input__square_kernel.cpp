#include <torch/extension.h>
#include <vector>
#include <omp.h>

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
    }

    // Convert CUDA tensors to CPU tensors
    auto x_cpu = x.cpu();
    auto weight_cpu = weight.cpu();
    auto bias_cpu = bias.has_value() ? bias.value().cpu() : torch::Tensor();

    // Perform the convolution operation on the CPU
    if (bias.has_value()) {
        return torch::conv2d(x_cpu, weight_cpu, bias_cpu, {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
    } else {
        return torch::conv2d(x_cpu, weight_cpu, torch::Tensor(), {stride, stride}, {padding, padding}, {dilation, dilation}, groups);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CPU forward function for 2D convolution with optional bias");
}