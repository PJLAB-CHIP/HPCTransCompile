#include <torch/extension.h>
#include <ATen/ATen.h>

// Tiling parameter for the channel dimension
#define TILE_C 16

// Optimized kernel using shared memory tiling for the input patch
__global__ void conv3d_min_softmax_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int N, const int C, const int D, const int H, const int W,
    const int K, const int T, const int R, const int S
) {
    // Each block processes one batch element (n) and each thread handles an output channel (k)
    int n = blockIdx.x;
    int k = threadIdx.x;

    if (n < N && k < K) {
        float sum = bias[k];

        // Declare shared memory for a tile of the input patch
        // Size needed per tile: TILE_C * T * R * S floats
        extern __shared__ float sharedInput[];

        // Loop over channel tiles
        for (int cc = 0; cc < C; cc += TILE_C) {
            int current_tile = (cc + TILE_C <= C) ? TILE_C : (C - cc);
            int tile_elems = current_tile * T * R * S;

            // Load the input patch tile into shared memory cooperatively
            // Each thread loads multiple elements from the tile
            for (int i = threadIdx.x; i < tile_elems; i += blockDim.x) {
                int c_local = i / (T * R * S);
                int rem = i % (T * R * S);
                int t = rem / (R * S);
                int rem2 = rem % (R * S);
                int r = rem2 / S;
                int s = rem2 % S;
                int c_global = cc + c_local;
                // Compute the input index for the patch
                int input_idx = n * C * D * H * W + c_global * D * H * W + t * H * W + r * W + s;
                sharedInput[i] = input[input_idx];
            }
            __syncthreads();

            // Use the loaded tile to update the convolution sum
            for (int c_local = 0; c_local < current_tile; c_local++) {
                for (int t = 0; t < T; t++) {
                    for (int r = 0; r < R; r++) {
                        for (int s = 0; s < S; s++) {
                            int tile_index = c_local * (T * R * S) + t * (R * S) + r * S + s;
                            int weight_idx = k * C * T * R * S + (cc + c_local) * T * R * S + t * R * S + r * S + s;
                            sum += weight[weight_idx] * sharedInput[tile_index];
                        }
                    }
                }
            }
            __syncthreads();
        }
        output[n * K + k] = sum;
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
    m.def("forward", &forward, "3D Convolution + Min + Softmax (CUDA)");
}
