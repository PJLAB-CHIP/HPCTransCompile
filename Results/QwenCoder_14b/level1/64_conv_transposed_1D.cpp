#include <torch/extension.h>
#include <vector>
#include <omp.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor forward_cpu(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups) {
    
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(weight);

    // Get dimensions
    int64_t batch_size = x.size(0);
    int64_t input_channels = x.size(1);
    int64_t input_length = x.size(2);
    int64_t output_channels = weight.size(0);
    int64_t kernel_size = weight.size(2);

    // Calculate output length
    int64_t output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;

    // Initialize output tensor
    torch::Tensor output = torch::zeros({batch_size, output_channels, output_length}, x.options());

    // Perform transposed 1D convolution
    #pragma omp parallel for collapse(3)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t g = 0; g < groups; ++g) {
            for (int64_t oc = g; oc < output_channels; oc += groups) {
                for (int64_t ic = g; ic < input_channels; ic += groups) {
                    for (int64_t i = 0; i < input_length; ++i) {
                        for (int64_t k = 0; k < kernel_size; ++k) {
                            int64_t o = i * stride - padding + k;
                            if (o >= 0 && o < output_length) {
                                output[b][oc][o] += x[b][ic][i] * weight[oc][ic][k];
                            }
                        }
                    }
                }
            }
        }
    }

    // Add bias if provided
    if (bias.has_value()) {
        CHECK_CONTIGUOUS(bias.value());
        output += bias.value().unsqueeze(2).expand_as(output);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "Transposed 1D convolution forward (CPU)");
}
