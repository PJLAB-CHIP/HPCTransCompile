#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void optimized_maxpool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N, const int C,
    const int D1, const int H1, const int W1,  // Dimensions after conv_transpose
    const int D3, const int H3, const int W3)  // Final dimensions
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * D3 * H3 * W3) return;

    // Decode output index
    const int w3 = idx % W3;
    const int h3 = (idx / W3) % H3;
    const int d3 = (idx / (W3 * H3)) % D3;
    const int c = (idx / (W3 * H3 * D3)) % C;
    const int n = idx / (W3 * H3 * D3 * C);

    // Calculate starting indices for the 3x3x3 window in the first maxpool output
    const int start_d2 = d3 * 3;
    const int start_h2 = h3 * 3;
    const int start_w2 = w3 * 3;

    float final_max = -FLT_MAX;

    // Use a single loop to minimize divergence
    for (int offset = 0; offset < 27; offset++) {
        int d2_offset = offset / 9;
        int h2_offset = (offset / 3) % 3;
        int w2_offset = offset % 3;

        const int d2 = start_d2 + d2_offset;
        const int h2 = start_h2 + h2_offset;
        const int w2 = start_w2 + w2_offset;

        // Check bounds collectively to minimize divergence
        if (d2 < D1/2 && h2 < H1/2 && w2 < W1/2) {
            // For each position in the 3x3x3 window, compute 2x2x2 maxpool
            float local_max = -FLT_MAX;

            // Starting indices for the 2x2x2 window in the original input
            const int start_d1 = d2 * 2;
            const int start_h1 = h2 * 2;
            const int start_w1 = w2 * 2;

            // Unrolled 2x2x2 maxpool
            for (int sub_offset = 0; sub_offset < 8; sub_offset++) {
                int d1_offset = sub_offset / 4;
                int h1_offset = (sub_offset / 2) % 2;
                int w1_offset = sub_offset % 2;

                const int d1 = start_d1 + d1_offset;
                const int h1 = start_h1 + h1_offset;
                const int w1 = start_w1 + w1_offset;

                // Check bounds collectively
                if (d1 < D1 && h1 < H1 && w1 < W1) {
                    const int input_idx = ((n * C + c) * D1 + d1) * H1 * W1 + h1 * W1 + w1;
                    local_max = max(local_max, input[input_idx]);
                }
            }

            final_max = max(final_max, local_max);
        }
    }

    output[idx] = final_max;
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias) {

    x = x.contiguous();
    conv_transpose = conv_transpose.contiguous();
    conv_transpose_bias = conv_transpose_bias.contiguous();

    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(conv_transpose.is_cuda(), "conv_transpose must be a CUDA tensor");
    TORCH_CHECK(conv_transpose_bias.is_cuda(), "conv_transpose_bias must be a CUDA tensor");

    // Apply transposed convolution using ATen op
    x = at::conv_transpose3d(
        x,
        conv_transpose,
        conv_transpose_bias,
        {stride, stride, stride},
        {padding, padding, padding}
    );

    // Get dimensions after conv_transpose
    auto sizes = x.sizes();
    const int N = sizes[0];
    const int C = sizes[1];
    const int D1 = sizes[2];
    const int H1 = sizes[3];
    const int W1 = sizes[4];

    // Calculate final dimensions after combined maxpool
    const int D3 = D1 / 6;
    const int H3 = H1 / 6;
    const int W3 = W1 / 6;

    // Allocate output tensor
    auto output = torch::empty({N, C, D3, H3, W3}, x.options());

    // Launch kernel
    const int total_elements = N * C * D3 * H3 * W3;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    optimized_maxpool_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D1, H1, W1, D3, H3, W3
    );

    // Sum over channels
    return output.sum(1, /*keepdim=*/true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Forward pass with optimized max pooling");
}