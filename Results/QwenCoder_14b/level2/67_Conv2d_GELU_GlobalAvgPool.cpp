#include <torch/extension.h>
#include <cmath>
#include <omp.h>

#define KERNEL_SIZE 3

float gelu_activate(float x) {
    return 0.5f * x * (1.f + std::erff(x / 1.41421356f));
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias
) {
    const int N = input.size(0);
    const int in_channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_channels = conv_weight.size(0);
    const int out_h = in_h - 2;
    const int out_w = in_w - 2;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty({N, out_channels}, options);

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int c_out = 0; c_out < out_channels; ++c_out) {
            float thread_sum = 0.0f;

            for (int out_row = 0; out_row < out_h; ++out_row) {
                for (int out_col = 0; out_col < out_w; ++out_col) {
                    float conv_result = 0.0f;

                    for (int ic = 0; ic < in_channels; ++ic) {
                        const float* in_ptr = &input[n][ic][out_row][out_col];
                        const float* w_ptr = &conv_weight[c_out][ic];

                        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
                            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                                conv_result += in_ptr[kh * in_w + kw] * w_ptr[kh * KERNEL_SIZE + kw];
                            }
                        }
                    }

                    conv_result = gelu_activate(conv_result + conv_bias[c_out]);
                    thread_sum += conv_result;
                }
            }

            output[n][c_out] = thread_sum / float(out_h * out_w);
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized reduction Conv2d + GELU + GlobalAvgPool");
}