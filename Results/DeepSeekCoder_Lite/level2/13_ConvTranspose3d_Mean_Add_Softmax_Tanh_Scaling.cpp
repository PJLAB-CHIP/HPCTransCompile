#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

torch::Tensor forward_cpu(
    const torch::Tensor& input,
    const torch::Tensor& conv_weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& spatial_bias,
    float scaling_factor,
    int stride,
    int padding
) {
    TORCH_CHECK(input.dim() == 5, "Input must be 5D tensor");
    TORCH_CHECK(conv_weight.dim() == 5, "Conv weight must be 5D tensor");

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);

    const int out_channels = conv_weight.size(1);
    const int kernel_d = conv_weight.size(2);
    const int kernel_h = conv_weight.size(3);
    const int kernel_w = conv_weight.size(4);

    const int out_depth = (in_depth - 1) * stride + kernel_d - 2 * padding;
    const int out_height = (in_height - 1) * stride + kernel_h - 2 * padding;
    const int out_width = (in_width - 1) * stride + kernel_w - 2 * padding;

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    torch::Tensor output = torch::empty({batch_size, 1, out_depth, out_height, out_width}, options);

    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int in_d = 0; in_d < in_depth; ++in_d) {
                for (int in_h = 0; in_h < in_height; ++in_h) {
                            for (int in_w = 0; in_w < in_width; ++in_w) {
                                for (int oc = 0; oc < out_channels; ++oc) {
                                    for (int kd = 0; kd < kernel_d; ++kd) {
                                        for (int kh = 0; kh < kernel_h; ++kh) {
                                            for (int kw = 0; kw < kernel_w; ++kw) {
                                                float total = 0.0f;
                                                for (int ic2 = 0; ic2 < in_channels; ++ic2) {
                                                    for (int kd2 = 0; kd2 < kernel_d; ++kd2) {
                                                        for (int kh2 = 0; kh2 < kernel_h; ++kh2) {
                                                            for (int kw2 = 0; kw2 < kernel_w; ++kw2) {
                                                                int in_d_unclamped = (in_d - kd + padding) / stride;
                                                                int in_h_unclamped = (in_h - kh + padding) / stride;
                                                                int in_w_unclamped = (in_w - kw + padding) / stride;
                                                                bool stride_valid = 
                                                                    ((in_d - kd + padding) % stride == 0) &&
                                                                    ((in_h - kh + padding) % stride == 0) &&
                                                                    ((in_w - kw + padding) % stride == 0);
                                                                bool in_bounds = 
                                                                    (in_d_unclamped >= 0) && (in_d_unclamped < in_depth) &&
                                                                    (in_h_unclamped >= 0) && (in_h_unclamped < in_height) &&
                                                                    (in_w_unclamped >= 0) && (in_w_unclamped < in_width);
                                                                float valid = (stride_valid && in_bounds) ? 1.0f : 0.0f;
                                                                int in_d2 = std::max(0, std::min(in_depth - 1, in_d_unclamped));
                                                                int in_h2 = std::max(0, std::min(in_height - 1, in_h_unclamped));
                                                                int in_w2 = std::max(0, std::min(in_width - 1, in_w_unclamped));
                                                                int input_idx = (((b * in_channels + ic) * in_depth + in_d2)
                                                                                * in_height + in_h2) * in_width + in_w2;
                                                                int weight_idx = (((ic2 * out_channels + oc) * kernel_d + kd2)
                                                                                * kernel_h + kh2) * kernel_w + kw2;
                                                                total += input[b, ic, in_d2, in_h2, in_w2].item<float>() * conv_weight[b, ic2, kd2, kh2, kw2].item<float>() * valid;
                                                            }
                                                        }
                                                    }
                                                }
                                                int spatial_idx = (in_d * out_height * out_width) + (in_h * out_width) + in_w;
                                                float mean_val = total / out_channels;
                                                float biased = mean_val + spatial_bias[spatial_idx].item<float>();
                                                output[b, 0, in_d, in_h, in_w] = tanhf(1.0f) * scaling_factor;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "Fused Transposed Conv3D Operations (CPU)");
}