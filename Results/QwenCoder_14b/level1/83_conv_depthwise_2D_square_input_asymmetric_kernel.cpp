include <torch/extension.h>
#include <omp.h>
#include <stdexcept>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)
#define TILE_SIZE 32

float compute_conv_pixel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int b, int c, int oh, int ow,
    int in_h, int in_w, int channels,
    int kernel_h, int stride, int padding, int dilation) {
    
    float sum = 0.0f;
    for (int kh = 0; kh < kernel_h; ++kh) {
        int ih = oh * stride - padding + kh * dilation;
        int iw = ow * stride - padding;
        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
            int weight_idx = c * kernel_h + kh;
            sum += input[input_idx] * weight[weight_idx];
        }
    }
    return sum;
}

void hybrid_depthwise_conv2d_cpu(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int channels, int in_h, int in_w,
    int out_h, int out_w, int kernel_h,
    int stride, int padding, int dilation) {
    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = compute_conv_pixel(
                        input, weight, b, c, oh, ow,
                        in_h, in_w, channels, kernel_h,
                        stride, padding, dilation);
                    
                    sum += bias[c];
                    const int output_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    x = x.contiguous();
    weight = weight.contiguous();
    
    const int batch = x.size(0);
    const int channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int kernel_h = weight.size(2);
    
    if (groups != channels)
        throw std::invalid_argument("Depthwise convolution requires groups == channels");
    
    at::Tensor bias_val = bias.has_value() && bias.value().defined() 
        ? bias.value().contiguous() 
        : at::zeros({channels}, x.options());
    
    const int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    const int out_w = (in_w + 2 * padding - 1) / stride + 1;
    
    auto output = at::empty({batch, channels, out_h, out_w}, x.options());
    
    hybrid_depthwise_conv2d_cpu(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_val.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, in_h, in_w,
        out_h, out_w, kernel_h,
        stride, padding, dilation
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise convolution forward");
}