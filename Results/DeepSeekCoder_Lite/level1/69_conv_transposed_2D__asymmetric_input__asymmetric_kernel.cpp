#include <torch/extension.h>
#include <vector>
#include <omp.h>

torch::Tensor conv_transpose2d_cpu(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> dilation,
    int64_t groups) {

    // Ensure the input tensors are in the correct device and type
    TORCH_CHECK(x.device().is_cpu(), "Input tensor must be on CPU");
    TORCH_CHECK(weight.device().is_cpu(), "Weight tensor must be on CPU");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be of type Float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight tensor must be of type Float32");

    // Get the dimensions of the tensors
    int64_t batch_size = x.size(0);
    int64_t in_channels = x.size(1);
    int64_t height = x.size(2);
    int64_t width = x.size(3);
    int64_t out_channels = weight.size(0);
    int64_t kernel_size = weight.size(2);
    int64_t kernel_size_h = kernel_size;
    int64_t kernel_size_w = kernel_size;

    // Calculate the output size
    int64_t out_height = (height - 1) * stride[0] - 2 * padding[0] + kernel_size_h + output_padding[0];
    int64_t out_width = (width - 1) * stride[1] - 2 * padding[1] + kernel_size_w + output_padding[1];

    // Create the output tensor
    torch::Tensor output = torch::empty({batch_size, out_channels, out_height, out_width}, x.options());

    // Check if bias is provided and has the correct size
    torch::Tensor bias_tensor = bias.value_or(torch::Tensor());
    TORCH_CHECK(bias_tensor.numel() == out_channels, "Bias tensor must have the same number of elements as the number of output channels");

    // Perform the convolution transpose operation
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t oc = 0; oc < out_channels; ++oc) {
            for (int64_t oh = 0; oh < out_height; ++oh) {
                for (int64_t ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0;
                    for (int64_t ic = 0; ic < in_channels; ++ic) {
                        for (int64_t kh = 0; kh < kernel_size_h; ++kh) {
                            for (int64_t kw = 0; kw < kernel_size_w; ++kw) {
                                int64_t ih = oh + kh * dilation[0] - padding[0];
                                int64_t iw = ow + kw * dilation[1] - padding[1];
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    sum += x[b][ic][ih][iw] * weight[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                    if (bias_tensor.defined()) {
                        sum += bias_tensor[oc];
                    }
                    output[b][oc][oh][ow] = sum;
                }
            }
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cpu, "ConvTranspose2D forward (CPU)");
}