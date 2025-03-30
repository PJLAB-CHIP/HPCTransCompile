#include <torch/extension.h>
#include <omp.h>

// Forward function definition
torch::Tensor conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {

    // Ensure inputs are on CUDA
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias tensor must be on CUDA");
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous");
    }

    // Use the built-in conv_transpose2d function for the main computation
    auto output = at::conv_transpose2d(
        x,
        weight,
        bias,
        {stride, stride},                   // stride
        {padding, padding},                 // padding
        {output_padding, output_padding},   // output_padding
        groups
    );

    // If bias is provided, add it using a separate kernel
    if (bias.has_value()) {
        int N = x.size(0);
        int C_out = weight.size(1);
        int H_out = output.size(2);
        int W_out = output.size(3);
        int total_output = N * C_out * H_out * W_out;

        #pragma omp parallel for
        for (int i = 0; i < total_output; i++) {
            int oc = (i / (H_out * W_out)) % C_out;
            output[i] += bias[oc];
        }
    }

    return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_forward, "ConvTranspose2d forward (CPU) - warp optimized");
}