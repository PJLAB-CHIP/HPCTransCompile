#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

__inline double mish_activation(double x) {
    return x * tanh(log(1.0 + exp(x)));
}

__inline float mish_activation(float x) {
    return x * tanh(log(1.0 + exp(x)));
}

template <typename scalar_t, bool IS_3x3 = true>
void conv2d_mish_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int k_h,
    const int k_w,
    const int out_h,
    const int out_w) {

    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    scalar_t sum = bias[oc];

                    if constexpr (IS_3x3) {
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kh = 0; kh < k_h; ++kh) {
                                for (int kw = 0; kw < k_w; ++kw) {
                                    int in_idx = b * (in_channels * in_h * in_w) + ic * (in_h * in_w) + (oh + kh) * in_w + ow + kw;
                                    int w_idx = (b * out_channels + oc) * (k_h * k_w) + (ic * k_h + kh) * k_w + kw;
                                    sum += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    } else {
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kh = 0; kh < k_h; ++kh) {
                                for (int kw = 0; kw < k_w; ++kw) {
                                    int in_idx = b * (in_channels * in_h * in_w) + ic * (in_h * in_w) + (oh + kh) * in_w + ow + kw;
                                    int w_idx = (b * out_channels + oc) * (k_h * k_w) + (ic * k_h + kh) * k_w + kw;
                                    sum += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }

                    output[(b * out_channels + oc) * (out_h * out_w) + oh * out_w + ow] = mish_activation(mish_activation(sum));
                }
            }
        }
    }
}

torch::Tensor forward(torch::Tensor input, torch::Tensor conv_weight, torch::Tensor conv_bias) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_h = input.size(2);
    const auto in_w = input.size(3);
    const auto out_channels = conv_weight.size(0);
    const auto k_h = conv_weight.size(2);
    const auto k_w = conv_weight.size(3);
    const auto out_h = in_h - k_h + 1;
    const auto out_w = in_w - k_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv2d_mish_forward", ([&] {
        conv2d_mish_kernel<scalar_t, true>(
            input.data_ptr<scalar_t>(),
            conv_weight.data_ptr<scalar_t>(),
            conv_bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_h,
            in_w,
            out_channels,
            k_h,
            k_w,
            out_h,
            out_w);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "2D Convolution with double Mish activation (CPU)");
}