#include <torch/extension.h>
#include <ATen/ATen.h>
#include <omp.h>

at::Tensor forward(
    const at::Tensor& x,
    int64_t stride,
    int64_t padding,
    const at::Tensor& conv_transpose,
    const at::Tensor& conv_transpose_bias,
    const at::Tensor& scale1,
    const at::Tensor& scale2,
    const at::Tensor& bias
) {
    // Check inputs
    TORCH_CHECK(x.dim() == 5, "Input tensor must have 5 dimensions");
    TORCH_CHECK(conv_transpose.dim() == 5, "Conv_transpose tensor must have 5 dimensions");
    TORCH_CHECK(conv_transpose_bias.dim() == 1, "Conv_transpose_bias tensor must have 1 dimension");
    TORCH_CHECK(scale1.dim() == 1, "Scale1 tensor must have 1 dimension");
    TORCH_CHECK(scale2.dim() == 1, "Scale2 tensor must have 1 dimension");
    TORCH_CHECK(bias.dim() == 1, "Bias tensor must have 1 dimension");

    // Get device and dtype
    auto device = x.device().is_cpu() ? x.device() : x.device().cpu();
    auto dtype = x.dtype();

    // Allocate output tensor
    auto output = at::empty({x.size(0), x.size(1), x.size(2) / 2, x.size(3) / 2, x.size(4) / 2}, x.options());

    // Perform transposed convolution
    auto y = at::conv_transpose3d(x, conv_transpose, conv_transpose_bias, {stride, stride, stride}, {padding, padding, padding});

    // Multiply by scale1
    y = y * scale1;

    // Perform average pooling with kernel_size=2
    auto pool_size = at::tensor({2, 2, 2}, y.options());
    y = at::avg_pool3d(y, pool_size);

    // Add bias
    y = y + bias;

    // Multiply by scale2
    y = y * scale2;

    // Copy result to output tensor
    output.copy_(y);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "module_fn forward (CPU)");
}