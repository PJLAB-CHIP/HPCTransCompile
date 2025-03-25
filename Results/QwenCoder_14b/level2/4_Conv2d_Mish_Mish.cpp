#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

inline float mish_activation(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

inline double mish_activation(double x) {
    return x * tanh(log(1.0 + exp(x)));
}

template <typename scalar_t, bool IS_3x3=true>
void conv2d_mish_cpu(
    const scalar_t* input,
    const scalar_t* weight,
    const scalar_t* bias,
    scalar_t* output,
    const int batch_size,
    const int in_channels,
    const int in_h,
    const int in_w,
    const int out_channels,
    const int k_h,
    const int k_w,
    const int out_h,
    const int out_w) {

    #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    const int out_idx = b * (out_channels * out_h * out_w) +
                                       oc * (out_h * out_w) +
                                       oh * out_w +
                                       ow;

                    scalar_t sum = bias[oc];

                    if constexpr (IS_3x3) {
                        const int batch_offset = b * (in_channels * in_h * in_w);
                        const int weight_oc_offset = oc * (in_channels * 9);

                        for (int ic = 0; ic < in_channels; ++ic) {
                            const scalar_t* in_ptr = input + batch_offset + ic * in_h * in_w + oh * in_w + ow;
                            const scalar_t* w_ptr = weight + weight_oc_offset + ic * 9;

                            scalar_t in_vals[9];
                            for (int i = 0; i < 3; ++i) {
                                in_vals[i*3 + 0] = in_ptr[i*in_w + 0];
                                in_vals[i*3 + 1] = in_ptr[i*in_w + 1];
                                in_vals[i*3 + 2] = in_ptr[i*in_w + 2];
                            }

                            for (int i = 0; i < 9; ++i) {
                                sum += in_vals[i] * w_ptr[i];
                            }
                        }
                    } else {
                        const int batch_offset = b * (in_channels * in_h * in_w);
                        const int weight_oc_offset = oc * (in_channels * k_h * k_w);

                        for (int ic = 0; ic < in_channels; ++ic) {
                            const int in_ch_offset = batch_offset + ic * in_h * in_w;
                            const int weight_ic_offset = weight_oc_offset + ic * k_h * k_w;

                            for (int kh = 0; kh < k_h; ++kh) {
                                const scalar_t* in_row = input + in_ch_offset + (oh + kh) * in_w + ow;
                                const scalar_t* w_row = weight + weight_ic_offset + kh * k_w;

                                for (int kw = 0; kw < k_w; ++kw) {
                                    sum += in_row[kw] * w_row[kw];
                                }
                            }
                        }
                    }

                    output[out_idx] = mish_activation(mish_activation(sum));
                }
            }
        }
    }
}

at::Tensor forward(at::Tensor input, at::Tensor conv_weight, at::Tensor conv_bias) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_h = input.size(2);
    const auto in_w = input.size(3);
    const auto out_channels = conv_weight.size(0);
    const auto k_h = conv_weight.size(2);
    const auto k_w = conv_weight.size(3);
    const auto out_h = in_h - k_h + 1;
    const auto out_w = in_w - k_w + 1;

    auto output = at::empty({batch_size, out_channels, out_h, out_w}, input.options());

    if (k_h == 3 && k_w == 3) {
        conv2d_mish_cpu<float, true>(
            input.data_ptr<float>(),
            conv_weight.data_ptr<float>(),
            conv_bias.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            in_h,
            in_w,
            out_channels,
            k_h,
            k_w,
            out_h,
            out_w);
    } else {
        conv2d_mish_cpu<float, false>(
            input.data_ptr<float>(),
            conv_weight.data_ptr<float>(),
            conv_bias.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            in_h,
            in_w,
            out_channels,
            k_h,
            k_w,
            out_h,
            out_w);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "2D Convolution with double Mish activation (CPU)");
}