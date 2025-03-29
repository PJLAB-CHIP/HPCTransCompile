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

    // Get the dimensions of the input tensor
    auto N = x.size(0);
    auto C_in = x.size(1);
    auto H_in = x.size(2);
    auto W_in = x.size(3);

    // Get the dimensions of the weight tensor
    auto C_out = weight.size(0);
    auto C_per_group = C_out / groups;
    auto K_h = weight.size(2);
    auto K_w = weight.size(3);

    // Calculate the output dimensions
    auto H_out = (H_in - 1) * stride[0] - 2 * padding[0] + K_h + output_padding[0];
    auto W_out = (W_in - 1) * stride[1] - 2 * padding[1] + K_w + output_padding[1];

    // Initialize the output tensor
    auto output = torch::zeros({N, C_out, H_out, W_out}, x.options());

    // Perform the convolution transpose operation
    #pragma omp parallel for collapse(4)
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups; ++g) {
            for (int c_out = g * C_per_group; c_out < (g + 1) * C_per_group; ++c_out) {
                for (int h_out = 0; h_out < H_out; ++h_out) {
                    for (int w_out = 0; w_out < W_out; ++w_out) {
                        int h_in = (h_out - output_padding[0] + padding[0] - K_h + 1) / stride[0];
                        int w_in = (w_out - output_padding[1] + padding[1] - K_w + 1) / stride[1];
                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            output[n][c_out][h_out][w_out] += x[n][g * C_in / groups + (c_out % C_per_group)][h_in][w_in] * weight[c_out][0][h_out - h_in * stride[0]][w_out - w_in * stride[1]];
                        }
                    }
                }
            }
        }
    }

    // Add the bias if it is provided
    if (bias.has_value()) {
        output += bias.value();
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose2d_cpu, "ConvTranspose2D forward (CPU)");
}