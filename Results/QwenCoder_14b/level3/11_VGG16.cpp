#include <torch/extension.h>
#include <vector>
#include <cmath>

void conv2d_cpu(const float* input, const float* weight, const float* bias,
                 float* output, int N, int C, int H, int W,
                 int K, int P, int stride) {
    #pragma omp parallel for collapse(3)
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    float sum = bias[k];
                    for (int c = 0; c < C; ++c) {
                        for (int kh = 0; kh < 3; ++kh) {
                            for (int kw = 0; kw < 3; ++kw) {
                                int ih = h - P + kh;
                                int iw = w - P + kw;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    float in_val = input[n * C * H * W + c * H * W + ih * W + iw];
                                    float weight_val = weight[k * C * 3 * 3 + c * 3 * 3 + kh * 3 + kw];
                                    sum += in_val * weight_val;
                                }
                            }
                        }
                    }
                    output[n * K * H * W + k * H * W + h * W + w] = sum;
                }
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_cpu, "Convolutional 2D CPU");
}