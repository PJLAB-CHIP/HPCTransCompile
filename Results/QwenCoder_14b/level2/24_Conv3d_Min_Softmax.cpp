#include <torch/extension.h>
#include <ATen/ATen.h>
#include <omp.h>

// Tiling parameter for the channel dimension
#define TILE_C 16

void conv3d_min_softmax_cpu(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int N, const int C, const int D, const int H, const int W,
    const int K, const int T, const int R, const int S
) {
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            float sum = bias[k];

            // Loop over channel tiles
            for (int cc = 0; cc < C; cc += TILE_C) {
                int current_tile = (cc + TILE_C <= C) ? TILE_C : (C - cc);

                // Use the loaded tile to update the convolution sum
                for (int c_local = 0; c_local < current_tile; ++c_local) {
                    for (int t = 0; t < T; ++t) {
                        for (int r = 0; r < R; ++r) {
                            for (int s = 0; s < S; ++s) {
                                int c_global = cc + c_local;
                                // Compute the input index for the patch
                                int input_idx = n * C * D * H * W + c_global * D * H * W + t * H * W + r * W + s;
                                int weight_idx = k * C * T * R * S + (cc + c_local) * T * R * S + t * R * S + r * S + s;
                                sum += weight[weight_idx] * input[input_idx];
                            }
                        }
                    }
                }
            }
            output[n * K + k] = sum;
        }
    }
}

at::Tensor forward(
    at::Tensor x,
    int64_t dim,
    at::Tensor conv_weight,
    at::Tensor conv_bias
) {
    // 1) 3D convolution with unrolled kernel
    auto y = at::conv3d(x, conv_weight, conv_bias);
    // 2) Min along the specified dimension
    y = std::get<0>(y.min(dim));
    // 3) Softmax along the channel dimension (dim=1)
    y = at::softmax(y, 1);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Convolution + Min + Softmax (CPU)");
}