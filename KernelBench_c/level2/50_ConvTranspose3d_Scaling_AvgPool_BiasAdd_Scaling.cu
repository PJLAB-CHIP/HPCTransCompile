#include <torch/extension.h>
#include <ATen/ATen.h>

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
    // Transposed convolution
    auto y = at::conv_transpose3d(
        x,
        conv_transpose,
        conv_transpose_bias,
        /*stride=*/{stride, stride, stride},
        /*padding=*/{padding, padding, padding}
    );

    // Multiply by scale1
    y = y * scale1;

    // Average Pooling with kernel_size=2
    y = at::avg_pool3d(
        y,
        /*kernel_size=*/{2, 2, 2}
    );

    // Add bias
    y = y + bias;

    // Multiply by scale2
    y = y * scale2;

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "module_fn forward (CUDA)");
}