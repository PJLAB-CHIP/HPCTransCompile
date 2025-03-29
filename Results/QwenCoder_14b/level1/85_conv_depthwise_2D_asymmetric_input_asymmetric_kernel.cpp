#include <torch/extension.h>
#include <omp.h>

// Define the CPU version of the depthwise convolution kernel
void combined_depthwise_conv2d_cpu(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int channels_per_group
) {
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; ++b) {
        for (int g = 0; g < groups; ++g) {
            for (int m = 0; m < channels_per_group; ++m) {
                int c = g * channels_per_group + m;
                for (int out_y = 0; out_y < out_h; ++out_y) {
                    for (int out_x = 0; out_x < out_w; ++out_x) {
                        float result = 0.0f;
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                int in_y = out_y * stride_h + kh * dilation_h - padding_h;
                                int in_x = out_x * stride_w + kw * dilation_w - padding_w;
                                if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                                    result += input[((b * in_channels + g) * in_h + in_y) * in_w + in_x] *
                                              weight[((g * channels_per_group + m) * kernel_h + kh) * kernel_w + kw];
                                }
                            }
                        }
                        if (bias != nullptr) {
                            result += bias[c];
                        }
                        output[((b * out_channels + c) * out_h + out_y) * out_w + out_x] = result;
                    }
                }
            }
        }
    }
}

// Host forward function
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    TORCH_CHECK(!x.device().is_cuda(), "x must be a CPU tensor");
    TORCH_CHECK(!weight.device().is_cuda(), "weight must be a CPU tensor");
    if (bias.has_value()) {
        TORCH_CHECK(!bias->device().is_cuda(), "bias must be a CPU tensor");
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);

    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_channels = groups * weight.size(1);
    int channels_per_group = out_channels / groups;

    // Compute output dimensions
    int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias->data_ptr<float>();
    }

    combined_depthwise_conv2d_cpu(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        channels_per_group
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined Depthwise Conv2D forward (CPU) with OpenMP parallelization");
}