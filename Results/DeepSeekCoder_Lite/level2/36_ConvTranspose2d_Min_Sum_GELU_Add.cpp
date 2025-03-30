#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <cfloat>
#include <cmath>
#include <omp.h>

#define WARP_SIZE 32
#define TILE_SIZE 32

torch::Tensor forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    torch::Tensor bias
) {
    x = x.contiguous();
    x = at::conv_transpose2d(x, conv_transpose, conv_transpose_bias,
        {stride, stride}, {padding, padding}, {output_padding, output_padding}, 1, {1, 1});
    
    auto sizes = x.sizes();
    int N = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];
    
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::zeros({N, 1, 1, W}, options);
    
    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float min_val = FLT_MAX;
                for (int c = 0; c < C; ++c) {
                    int index = ((n * C + c) * H + h) * W + w;
                    float val = x[index];
                    min_val = std::min(min_val, val);
                }
                output[n][0][0][w] = min_val;
            }
        }
    }
    
    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        for (int w = 0; w < W; ++w) {
            float sum = 0.0f;
            for (int th = 0; th < TILE_SIZE && (h + th) < H; ++th) {
                for (int tw = 0; tw < TILE_SIZE && (w + tw) < W; ++tw) {
                    sum += output[n][0][0][w];
                }
            }
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                sum += sum;
            }
            output[n][0][0][w] = sum;
        }
    }
    
    output = at::gelu(output);
    output = output + bias;
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused min-sum reduction forward");
}