#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <omp.h>

namespace py = pybind11;

template <typename scalar_t>
void depthwise_conv2d_cpu(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch,
    int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k,
    int stride,
    int padding,
    int dilation) {

    #pragma omp parallel for collapse(4)
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    scalar_t sum = 0;
                    for (int i = 0; i < k; ++i) {
                        for (int j = 0; j < k; ++j) {
                            const int ih = oh * stride - padding + i * dilation;
                            const int iw = ow * stride - padding + j * dilation;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                sum += input[n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw] *
                                       weight[c * k * k + i * k + j];
                            }
                        }
                    }
                    if (bias != nullptr) {
                        sum += bias[c];
                    }
                    output[n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = sum;
                }
            }
        }
    }
}

template <typename scalar_t>
void pointwise_conv2d_cpu(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width) {

    #pragma omp parallel for collapse(4)
    for (int n = 0; n < batch; ++n) {
        for (int out_ch = 0; out_ch < out_channels; ++out_ch) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    scalar_t sum = 0;
                    const int spatial_offset = h * width + w;
                    const int weight_offset = out_ch * in_channels;
                    const int input_batch_offset = n * in_channels * height * width;
                    
                    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                        sum += input[input_batch_offset + in_ch * height * width + spatial_offset] *
                               weight[weight_offset + in_ch];
                    }
                    
                    if (bias != nullptr) {
                        sum += bias[out_ch];
                    }
                    
                    output[n * out_channels * height * width +
                           out_ch * height * width +
                           spatial_offset] = sum;
                }
            }
        }
    }
}

torch::Tensor forward_cpu(
    const torch::Tensor& x,
    const torch::Tensor& depthwise_weight,
    const torch::Tensor& pointwise_weight,
    const torch::Tensor& depthwise_bias,
    const torch::Tensor& pointwise_bias,
    int stride,
    int padding,
    int dilation) {

    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int k = depthwise_weight.size(2);
    const int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    const int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

    auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_cpu", ([&] {
        depthwise_conv2d_cpu<scalar_t>(
            x.data_ptr<scalar_t>(),
            depthwise_weight.data_ptr<scalar_t>(),
            depthwise_bias.defined() ? depthwise_bias.data_ptr<scalar_t>() : nullptr,
            depthwise_output.data_ptr<scalar_t>(),
            batch, in_channels, in_h, in_w, out_h, out_w,
            k, stride, padding, dilation);
    }));

    const int out_channels = pointwise_weight.size(0);
    auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cpu", ([&] {
        pointwise_conv2d_cpu<scalar_t>(
            depthwise_output.data_ptr<scalar_t>(),
            pointwise_weight.data_ptr<scalar_t>(),
            pointwise_bias.defined() ? pointwise_bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch, in_channels, out_channels, out_h, out_w);
    }));

    return output;
}

at::Tensor toTensor(const py::object& obj) {
    if (obj.is_none()) return at::Tensor();
    try {
        return obj.cast<at::Tensor>();
    } catch (const py::cast_error& e) {
        if (py::hasattr(obj, "data")) {
            return obj.attr("data").cast<at::Tensor>();
        }
        throw std::runtime_error("Expected a torch Tensor or Parameter.");
    }
}

at::Tensor forward_wrapper(
    py::object x_obj,
    py::object depthwise_weight_obj,
    py::object pointwise_weight_obj,
    py::object depthwise_bias_obj,
    py::object pointwise_bias_obj,
    int stride,
    int padding,
    int dilation) {

    return forward_cpu(
        toTensor(x_obj),
        toTensor(depthwise_weight_obj),
        toTensor(pointwise_weight_obj),
        toTensor(depthwise_bias_obj),
        toTensor(pointwise_bias_obj),
        stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_wrapper, "CPU depthwise separable convolution forward");
}
