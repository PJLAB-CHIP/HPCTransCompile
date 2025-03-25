#include <torch/extension.h>
#include <omp.h>
#include <cmath>
#include <limits>

void strided_maxpool_cpu(
    const float* input,
    float* output,
    const int N, const int C, const int D, const int H, const int W,
    const int outD, const int outH, const int outW,
    const int items_per_block
) {
    // Pre-compute strides for more efficient indexing
    const int stride_n = C * D * H * W;
    const int stride_c = D * H * W;
    const int stride_d = H * W;
    const int stride_h = W;
    
    const int out_stride_c = outD * outH * outW;
    const int out_stride_d = outH * outW;
    
    #pragma omp parallel for collapse(5)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int out_d = 0; out_d < outD; ++out_d) {
                for (int out_h = 0; out_h < outH; ++out_h) {
                    for (int out_w = 0; out_w < outW; ++out_w) {
                        // Calculate input window start position
                        const int d_start = out_d * 4;
                        const int h_start = out_h * 4;
                        const int w_start = out_w * 4;

                        float max_val = -std::numeric_limits<float>::max();
                        for (int local_d = 0; local_d < 4; ++local_d) {
                            for (int local_h = 0; local_h < 4; ++local_h) {
                                for (int local_w = 0; local_w < 4; ++local_w) {
                                    const int d = d_start + local_d;
                                    const int h = h_start + local_h;
                                    const int w = h_start + local_w;

                                    if (d < D && h < H && w < W) {
                                        const int input_idx = n * stride_n + c * stride_c + d * stride_d + h * stride_h + w;
                                        max_val = std::max(max_val, input[input_idx]);
                                    }
                                }
                            }
                        }

                        const int out_idx = n * (C * outD * outH * outW) + 
                                          c * (outD * outH * outW) + 
                                          out_d * (outH * outW) + 
                                          out_h * outW + 
                                          out_w;
                        output[out_idx] = max_val;
                    }
                }
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias
) {
    x = x.contiguous();
    conv_weight = conv_weight.contiguous();
    conv_bias = conv_bias.contiguous();

    auto conv_output = at::conv3d(x, conv_weight, conv_bias, {1, 1, 1}, {0, 0, 0});
    auto softmax_output = at::softmax(conv_output, /*dim=*/1);

    const int N = softmax_output.size(0);
    const int C = softmax_output.size(1);
    const int D = softmax_output.size(2);
    const int H = softmax_output.size(3);
    const int W = softmax_output.size(4);

    const int outD = D / 4;
    const int outH = H / 4;
    const int outW = W / 4;

    auto options = softmax_output.options();
    auto output = torch::empty({N, C, outD, outH, outW}, options);

    // Calculate optimal items per block based on CPU characteristics
    const int items_per_block = 4;  // Process 4 output elements per block
    const int total_elements = N * C * outD * outH * outW;

    strided_maxpool_cpu(
        softmax_output.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        outD, outH, outW,
        items_per_block
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided CPU forward function");
}