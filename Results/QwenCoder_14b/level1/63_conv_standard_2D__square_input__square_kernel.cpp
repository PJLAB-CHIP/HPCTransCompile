#include <torch/extension.h>
#include <omp.h>
#include <vector>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void conv2d_cpu(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(weight);
    CHECK_CONTIGUOUS(output);

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

    #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch_size; ++b) {
        for (int g = 0; g < groups; ++g) {
            for (int oc = g * (out_channels / groups); oc < (g + 1) * (out_channels / groups); ++oc) {
                for (int oh = 0; oh < output_height; ++oh) {
                    for (int ow = 0; ow < output_width; ++ow) {
                        float sum = 0.0f;
                        for (int ic = g * (in_channels / groups); ic < (g + 1) * (in_channels / groups); ++ic) {
                            for (int kh = 0; kh < kernel_height; ++kh) {
                                for (int kw = 0; kw < kernel_width; ++kw) {
                                    int ih = oh * stride + kh * dilation - padding;
                                    int iw = ow * stride + kw * dilation - padding;
                                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                        sum += input[b][ic][ih][iw] * weight[oc][ic][kh][kw];
                                    }
                                }
                            }
                        }
                        output[b][oc][oh][ow] = sum;
                    }
                }
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(weight);

    int out_channels = weight.size(0);
    int output_height = (x.size(2) + 2 * padding - weight.size(2)) / stride + 1;
    int output_width = (x.size(3) + 2 * padding - weight.size(3)) / stride + 1;
    torch::Tensor output = torch::zeros({x.size(0), out_channels, output_height, output_width}, x.options());

    conv2d_cpu(x, weight, output, stride, padding, dilation, groups);

    if (bias.has_value()) {
        CHECK_CONTIGUOUS(bias.value());
        output += bias.value().unsqueeze(0).unsqueeze(2).unsqueeze(3);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CPU forward function for 2D convolution with optional bias");
}