#include <torch/extension.h>
#include <cmath>
#include <omp.h>

// Function that ensures memory coalescing by processing contiguous spatial locations per (n, c) pair
template <typename scalar_t>
void coalesced_bias_subtract_tanh_cpu(
    scalar_t* output,
    const scalar_t* bias,
    const int64_t N,
    const int64_t C_out,
    const int64_t H_out,
    const int64_t W_out
) {
    int64_t spatial_size = H_out * W_out;

    #pragma omp parallel for collapse(2)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C_out; ++c) {
            int64_t base_idx = n * (C_out * spatial_size) + c * spatial_size;
            scalar_t bias_val = bias[c];

            for (int64_t i = 0; i < spatial_size; ++i) {
                int64_t idx = base_idx + i;
                output[idx] = tanh(output[idx] - bias_val);
            }
        }
    }
}

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
    // Perform the transposed convolution on CPU
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

    // Apply the CPU version of the kernel
    AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "coalesced_bias_subtract_tanh_cpu", ([&] {
        coalesced_bias_subtract_tanh_cpu<scalar_t>(
            output.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            N, C_out, H_out, W_out
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CPU forward function with coalesced memory access");
}