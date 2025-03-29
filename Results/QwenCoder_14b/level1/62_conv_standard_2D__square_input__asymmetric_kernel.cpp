#include <torch/extension.h>
#include <vector>
#include <omp.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

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
    if (bias.has_value()) {
        CHECK_CONTIGUOUS(bias.value());
    }

    // Extract dimensions
    int N = x.size(0); // batch size
    int C_in = x.size(1); // input channels
    int H_in = x.size(2); // input height
    int W_in = x.size(3); // input width

    int C_out = weight.size(0); // output channels
    int K_h = weight.size(2); // kernel height
    int K_w = weight.size(3); // kernel width

    int H_out = (H_in + 2 * padding - K_h) / stride + 1; // output height
    int W_out = (W_in + 2 * padding - K_w) / stride + 1; // output width

    // Initialize output tensor
    torch::Tensor output = torch::zeros({N, C_out, H_out, W_out}, x.options());

    // Perform convolution
    #pragma omp parallel for collapse(4)
    for (int n = 0; n < N; ++n) {
        for (int c_out = 0; c_out < C_out; ++c_out) {
            for (int h_out = 0; h_out < H_out; ++h_out) {
                for (int w_out = 0; w_out < W_out; ++w_out) {
                    int h_in_start = h_out * stride - padding;
                    int w_in_start = w_out * stride - padding;

                    float sum = 0.0f;
                    for (int c_in = 0; c_in < C_in; ++c_in) {
                        for (int k_h = 0; k_h < K_h; ++k_h) {
                            for (int k_w = 0; k_w < K_w; ++k_w) {
                                int h_in = h_in_start + k_h * dilation;
                                int w_in = w_in_start + k_w * dilation;

                                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                    sum += x[n][c_in][h_in][w_in] * weight[c_out][c_in][k_h][k_w];
                                }
                            }
                        }
                    }

                    output[n][c_out][h_out][w_out] = sum;
                }
            }
        }
    }

    // Add bias if present
    if (bias.has_value()) {
        output += bias.value();
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CPU 2D Convolution");
}
