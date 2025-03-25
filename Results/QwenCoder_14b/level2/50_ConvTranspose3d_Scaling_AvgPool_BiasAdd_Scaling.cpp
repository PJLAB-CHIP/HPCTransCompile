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
    // Ensure the input tensors are on the CPU
    auto x_cpu = x.to(at::kCPU);
    auto conv_transpose_cpu = conv_transpose.to(at::kCPU);
    auto conv_transpose_bias_cpu = conv_transpose_bias.to(at::kCPU);
    auto scale1_cpu = scale1.to(at::kCPU);
    auto scale2_cpu = scale2.to(at::kCPU);
    auto bias_cpu = bias.to(at::kCPU);

    // Transposed convolution
    auto y = at::conv_transpose3d(
        x_cpu,
        conv_transpose_cpu,
        conv_transpose_bias_cpu,
        /*stride=*/{stride, stride, stride},
        /*padding=*/{padding, padding, padding}
    );

    // Multiply by scale1
    y = y * scale1_cpu;

    // Average Pooling with kernel_size=2
    y = at::avg_pool3d(
        y,
        /*kernel_size=*/{2, 2, 2}
    );

    // Add bias
    y = y + bias_cpu;

    // Multiply by scale2
    y = y * scale2_cpu;

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "module_fn forward (CPU)");
}