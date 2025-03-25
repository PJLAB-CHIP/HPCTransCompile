#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <cfloat>
#include <omp.h>

template <int KERNEL_H, int KERNEL_W>
void fused_conv_pool_cpu(
    const float* x,
    const float* conv_weight,
    const float* conv_bias,
    const float* bias,
    float* out,
    const float scaling_factor,
    const int batch_size,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int pool_kernel_size,
    const int out_h,
    const int out_w,
    const int pooled_h,
    const int pooled_w
) {
    #pragma omp parallel for collapse(3)
    for (int n = 0; n < batch_size; ++n) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ph = 0; ph < pooled_h; ++ph) {
                for (int pw = 0; pw < pooled_w; ++pw) {
                    int conv_oh_start = ph * pool_kernel_size;
                    int conv_ow_start = pw * pool_kernel_size;

                    float max_val = -FLT_MAX;
                    for (int py = 0; py < pool_kernel_size; ++py) {
                        for (int px = 0; px < pool_kernel_size; ++px) {
                            int conv_oh = conv_oh_start + py;
                            int conv_ow = conv_ow_start + px;
                            if (conv_oh < out_h && conv_ow < out_w) {
                                float val = conv_bias[oc];
                                for (int ic = 0; ic < in_channels; ++ic) {
                                    int input_base = ((n * in_channels + ic) * in_h);
                                    int weight_base = ((oc * in_channels + ic) * KERNEL_H);
                                    for (int kh = 0; kh < KERNEL_H; ++kh) {
                                        for (int kw = 0; kw < KERNEL_W; ++kw) {
                                            int in_row = conv_oh + kh;
                                            int in_col = conv_ow + kw;
                                            int input_idx = (input_base + in_row) * in_w + in_col;
                                            int weight_idx = (weight_base + kh) * KERNEL_W + kw;
                                            val += x[input_idx] * conv_weight[weight_idx];
                                        }
                                    }
                                }
                                val = tanhf(val) * scaling_factor + bias[oc];
                                if (val > max_val) {
                                    max_val = val;
                                }
                            }
                        }
                    }
                    int out_idx = ((n * out_channels + oc) * pooled_h + ph) * pooled_w + pw;
                    out[out_idx] = max_val;
                }
            }
        }
    }
}

torch::Tensor forward_cpu(
    torch::Tensor x,
    double scaling_factor,
    int pool_kernel_size,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor bias
) {
    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);

    const int out_channels = conv_weight.size(0);
    const int kernel_h = conv_weight.size(2);
    const int kernel_w = conv_weight.size(3);

    const int out_h = in_h - kernel_h + 1;
    const int out_w = in_w - kernel_w + 1;

    const int pooled_h = out_h / pool_kernel_size;
    const int pooled_w = out_w / pool_kernel_size;

    auto options = x.options();
    auto pooled_out = torch::empty({batch_size, out_channels, pooled_h, pooled_w}, options);

    if (kernel_h == 3 && kernel_w == 3) {
        fused_conv_pool_cpu<3, 3>(
            x.data_ptr<float>(),
            conv_weight.data_ptr<float>(),
            conv_bias.data_ptr<float>(),
            bias.data_ptr<float>(),
            pooled_out.data_ptr<float>(),
            static_cast<float>(scaling_factor),
            batch_size, in_channels, in_h, in_w,
            out_channels,
            pool_kernel_size,
            out_h, out_w,
            pooled_h, pooled_w
        );
    } else if (kernel_h == 5 && kernel_w == 5) {
        fused_conv_pool_cpu<5, 5>(
            x.data_ptr<float>(),
            conv_weight.data_ptr<float>(),
            conv_bias.data_ptr<float>(),
            bias.data_ptr<float>(),
            pooled_out.data_ptr<float>(),
            static_cast<float>(scaling_factor),
            batch_size, in_channels, in_h, in_w,
            out_channels,
            pool_kernel_size,
            out_h, out_w,
            pooled_h, pooled_w
        );
    } else {
        throw std::runtime_error("Only 3x3 and 5x5 kernels are supported in the fused version");
    }

    return pooled_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "Fused conv-tanh-scale-add and max pool forward (CPU)");
}