#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <omp.h>
#include <algorithm>
#include <limits>

namespace py = pybind11;

void optimized_maxpool_cpu(
    const float* input,
    float* output,
    const int N, const int C,
    const int D1, const int H1, const int W1,  // Dimensions after conv_transpose
    const int D3, const int H3, const int W3)  // Final dimensions
{
    #pragma omp parallel for collapse(5)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int d3 = 0; d3 < D3; ++d3) {
                for (int h3 = 0; h3 < H3; ++h3) {
                    for (int w3 = 0; w3 < W3; ++w3) {
                        const int start_d2 = d3 * 3;
                        const int start_h2 = h3 * 3;
                        const int start_w2 = w3 * 3;

                        float final_max = -std::numeric_limits<float>::max();

                        for (int offset = 0; offset < 27; offset++) {
                            int d2_offset = offset / 9;
                            int h2_offset = (offset / 3) % 3;
                            int w2_offset = offset % 3;

                            const int d2 = start_d2 + d2_offset;
                            const int h2 = start_h2 + h2_offset;
                            const int w2 = start_w2 + w2_offset;

                            if (d2 < D1/2 && h2 < H1/2 && w2 < W1/2) {
                                float local_max = -std::numeric_limits<float>::max();

                                const int start_d1 = d2 * 2;
                                const int start_h1 = h2 * 2;
                                const int start_w1 = w2 * 2;

                                for (int sub_offset = 0; sub_offset < 8; sub_offset++) {
                                    int d1_offset = sub_offset / 4;
                                    int h1_offset = (sub_offset / 2) % 2;
                                    int w1_offset = sub_offset % 2;

                                    const int d1 = start_d1 + d1_offset;
                                    const int h1 = start_h1 + h1_offset;
                                    const int w1 = start_h1 + w1_offset;

                                    if (d1 < D1 && h1 < H1 && w1 < W1) {
                                        const int input_idx = ((n * C + c) * D1 + d1) * H1 * W1 + h1 * W1 + w1;
                                        local_max = std::max(local_max, input[input_idx]);
                                    }
                                }

                                final_max = std::max(final_max, local_max);
                            }
                        }

                        const int output_idx = ((n * C + c) * D3 + d3) * H3 * W3 + h3 * W3 + w3;
                        output[output_idx] = final_max;
                    }
                }
            }
        }
    }
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

    // Perform maxpool on CPU
    optimized_maxpool_cpu(
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