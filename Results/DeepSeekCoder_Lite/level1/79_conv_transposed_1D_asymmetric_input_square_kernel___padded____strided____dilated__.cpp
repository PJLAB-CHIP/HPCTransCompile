#include <torch/extension.h>
#include <omp.h>

// Helper function to compute the output length
inline int compute_output_length(int input_length, int stride, int padding, int dilation, int kernel_size) {
    return (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
}

// CPU implementation of the convolution transpose 1D operation
torch::Tensor forward_cpu(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation
) {
    TORCH_CHECK(x.device().is_cpu(), "x must be a CPU tensor");
    TORCH_CHECK(weight.device().is_cpu(), "weight must be a CPU tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (batch, in_channels, input_length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");

    x = x.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_contig;
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_contig = bias->contiguous();
        TORCH_CHECK(bias_contig.device().is_cpu(), "bias must be a CPU tensor");
        TORCH_CHECK(bias_contig.dim() == 1, "bias must be 1D");
        bias_ptr = bias_contig.data_ptr<float>();
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "weight's in_channels must match x's in_channels");
    if (bias.has_value()) {
        TORCH_CHECK(bias_contig.size(0) == out_channels, "bias size must match out_channels");
    }

    int output_length = compute_output_length(input_length, stride, padding, dilation, kernel_size);
    auto output = torch::zeros({batch_size, out_channels, output_length}, x.options());

    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int o = 0; o < output_length; ++o) {
                float sum = 0.0f;
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int k = 0; k < kernel_size; ++k) {
                        int i_pos = o + padding - k * dilation;
                        if (i_pos % stride != 0) continue;
                        int i = i_pos / stride;
                        if (i < 0 || i >= input_length) continue;
                        sum += x[b][ic][i] * weight[ic][oc][k];
                    }
                }
                if (bias_ptr) {
                    sum += bias_ptr[oc];
                }
                output[b][oc][o] = sum;
            }
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "ConvTranspose1D forward (CPU) with OpenMP optimization",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"));
}