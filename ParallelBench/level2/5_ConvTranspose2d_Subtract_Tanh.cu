#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")

// Kernel that ensures memory coalescing by processing contiguous spatial locations per (n, c) pair
template <typename scalar_t>
__global__ void coalesced_bias_subtract_tanh_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ bias,
    const int64_t N,
    const int64_t C_out,
    const int64_t H_out,
    const int64_t W_out
) {
    // Each block handles one (n, c) pair
    int64_t block_id = blockIdx.x;  
    int64_t n = block_id / C_out;
    int64_t c = block_id % C_out;
    int64_t spatial_size = H_out * W_out;
    int64_t base_idx = n * (C_out * spatial_size) + c * spatial_size;

    // Load bias value once into registers using read-only cache
    scalar_t bias_val = __ldg(&bias[c]);

    // Each thread processes multiple contiguous spatial elements to ensure coalesced accesses
    for (int64_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
         int64_t idx = base_idx + i;
         output[idx] = tanh(output[idx] - bias_val);
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
    CHECK_CUDA(x);
    CHECK_CUDA(conv_transpose);
    CHECK_CUDA(conv_transpose_bias);
    CHECK_CUDA(bias);

    torch::DeviceGuard guard(x.device());

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
    dim3 grid(N * C_out);
    const int threads = 256;
    
    AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "coalesced_bias_subtract_tanh_cuda", ([&] {
        coalesced_bias_subtract_tanh_kernel<scalar_t><<<grid, threads>>>(
            output.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            N, C_out, H_out, W_out
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA forward function with coalesced memory access");
}
