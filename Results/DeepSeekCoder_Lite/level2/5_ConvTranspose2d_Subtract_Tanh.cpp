#include <torch/extension.h>
#include <cmath>
#include <omp.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")

// Forward function: runs conv_transpose2d then applies bias subtraction and tanh activation
torch::Tensor forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    torch::Tensor bias
) {
    CHECK_CUDA(x);
    CHECK_CUDA(conv_transpose);
    CHECK_CUDA(conv_transpose_bias);
    CHECK_CUDA(bias);

    // Perform the transposed convolution
    auto output = at::conv_transpose2d(
        x,
        conv_transpose,
        conv_transpose_bias,
        {stride, stride},
        {padding, padding},
        {output_padding, output_padding},
        1  // groups
    );

    // Get output dimensions
    int64_t N = output.size(0);
    int64_t C_out = output.size(1);
    int64_t H_out = output.size(2);
    int64_t W_out = output.size(3);

    // Launch one block per (n, c) pair to ensure that spatial elements are contiguous
    #pragma omp parallel for
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C_out; ++c) {
            int64_t spatial_size = H_out * W_out;
            int64_t base_idx = n * (C_out * spatial_size) + c * spatial_size;
            scalar_t bias_val = bias[c];

            for (int64_t i = 0; i < spatial_size; ++i) {
                int64_t idx = base_idx + i;
                output[idx] = tanh(output[idx] - bias_val);
            }
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function with coalesced memory access");
}