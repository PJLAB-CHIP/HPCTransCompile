#include <torch/extension.h>
#include <vector>
#include <omp.h>

// Function to perform conv_transpose3d operation
torch::Tensor conv_transpose3d_cpu(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups
) {
    // Get input dimensions
    auto N = x.size(0);
    auto C_in = x.size(1);
    auto D_in = x.size(2);
    auto H_in = x.size(3);
    auto W_in = x.size(4);

    // Get weight dimensions
    auto C_out = weight.size(0);
    auto K_d = weight.size(2);
    auto K_h = weight.size(3);
    auto K_w = weight.size(4);

    // Calculate output dimensions
    auto D_out = (D_in - 1) * stride[0] - 2 * padding[0] + K_d + output_padding[0];
    auto H_out = (H_in - 1) * stride[1] - 2 * padding[1] + K_h + output_padding[1];
    auto W_out = (W_in - 1) * stride[2] - 2 * padding[2] + K_w + output_padding[2];

    // Initialize output tensor
    torch::Tensor output = torch::zeros({N, C_out, D_out, H_out, W_out}, x.options());

    // Perform convolution transpose operation
    #pragma omp parallel for collapse(5)
    for (int n = 0; n < N; ++n) {
        for (int c_out = 0; c_out < C_out; ++c_out) {
            for (int d_out = 0; d_out < D_out; ++d_out) {
                for (int h_out = 0; h_out < H_out; ++h_out) {
                    for (int w_out = 0; w_out < W_out; ++w_out) {
                        int d_in = (d_out - K_d + 2 * padding[0]) / stride[0];
                        int h_in = (h_out - K_h + 2 * padding[1]) / stride[1];
                        int w_in = (w_out - K_w + 2 * padding[2]) / stride[2];

                        if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            output[n][c_out][d_out][h_out][w_out] += x[n][d_in][h_in][w_in] * weight[c_out][0][d_out - d_in][h_out - h_in][w_out - w_in];
                        }
                    }
                }
            }
        }
    }

    // Add bias if provided
    if (bias.has_value()) {
        output += bias.value().unsqueeze(2).unsqueeze(3).unsqueeze(4);
    }

    return output;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_transpose3d_cpu, "ConvTranspose3d forward function",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = nullptr,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("groups"));
}