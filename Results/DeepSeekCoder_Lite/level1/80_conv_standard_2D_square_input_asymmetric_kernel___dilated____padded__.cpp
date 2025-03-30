#include <torch/extension.h>
#include <omp.h>

#define CHANNELS_PER_BLOCK 4

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,  // optional bias
    int stride,
    std::tuple<int, int> padding,
    std::tuple<int, int> dilation) {

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        bias_ptr = bias->data_ptr<float>();
    }
    
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_height = x.size(2);
    int input_width = x.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    int pad_h = std::get<0>(padding);
    int pad_w = std::get<1>(padding);
    int dilation_h = std::get<0>(dilation);
    int dilation_w = std::get<1>(dilation);

    int height_out = (input_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride + 1;
    int width_out = (input_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, height_out, width_out}, x.options());

    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        for (int g = 0; g < (out_channels + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK; ++g) {
            for (int h_out = 0; h_out < height_out; ++h_out) {
                for (int w_out = 0; w_out < width_out; ++w_out) {
                    float sums[CHANNELS_PER_BLOCK] = {0.0f, 0.0f, 0.0f, 0.0f};

                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            int h_in = h_out * stride + kh * dilation_h - pad_h;
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                int w_in = w_out * stride + kw * dilation_w - pad_w;
                                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                                    for (int i = 0; i < CHANNELS_PER_BLOCK; ++i) {
                                        int global_oc = g * CHANNELS_PER_BLOCK + i;
                                        if (global_oc < out_channels) {
                                            for (int idx = 0; idx < CHANNELS_PER_BLOCK; ++idx) {
                                                int weight_offset = idx * (in_channels * kernel_h * kernel_w) +
                                                                    ic * (kernel_h * kernel_w) +
                                                                    kh * kernel_w + kw;
                                                sums[idx] += x[b * in_channels * input_height * input_width +
                                                                 ic * input_height * input_width +
                                                                 h_in * input_width + w_in] *
                                                             weight[global_oc * in_channels * kernel_h * kernel_w + weight_offset];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for (int i = 0; i < CHANNELS_PER_BLOCK; ++i) {
                        int global_oc = g * CHANNELS_PER_BLOCK + i;
                        if (global_oc < out_channels) {
                            int out_idx = b * out_channels * height_out * width_out +
                                          global_oc * height_out * width_out +
                                          h_out * width_out + w_out;
                            output[out_idx] = sums[i] + (bias_ptr ? bias_ptr[global_oc] : 0.0f);
                        }
                    }
                }
            }
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Conv2D forward (CPU)");
}