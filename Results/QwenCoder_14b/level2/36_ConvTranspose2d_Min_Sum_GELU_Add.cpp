#include <torch/extension.h>
#include <vector>
#include <omp.h>
#include <limits>

#define TILE_SIZE 32

void fused_min_sum_cpu(
    const float* input,
    float* output,
    int N, int C, int H, int W
) {
    #pragma omp parallel for collapse(3)
    for (int n = 0; n < N; ++n) {
        for (int tile_h = 0; tile_h < H; tile_h += TILE_SIZE) {
            for (int tile_w = 0; tile_w < W; tile_w += TILE_SIZE) {
                float shared_min[TILE_SIZE][TILE_SIZE];
                for (int th = 0; th < TILE_SIZE && (tile_h + th) < H; ++th) {
                    for (int tw = 0; tw < TILE_SIZE && (tile_w + tw) < W; ++tw) {
                        float min_val = std::numeric_limits<float>::max();
                        for (int c = 0; c < C; ++c) {
                            float val = input[((n * C + c) * H + (tile_h + th)) * W + (tile_w + tw)];
                            min_val = std::min(min_val, val);
                        }
                        shared_min[th][tw] = min_val;
                    }
                }

                float sum = 0.0f;
                for (int th = 0; th < TILE_SIZE && (tile_h + th) < H; ++th) {
                    sum += shared_min[th][0];
                }

                output[(n * W) + tile_w] = sum;
            }
        }
    }
}

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
    
    fused_min_sum_cpu(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W
    );
    
    output = at::gelu(output);
    output = output + bias;
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused min-sum reduction forward");
}