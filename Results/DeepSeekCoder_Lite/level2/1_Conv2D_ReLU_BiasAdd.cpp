#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <omp.h>

torch::Tensor conv2d_relu_bias_forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor bias)
{
    TORCH_CHECK(x.is_cpu(), "x must be a CPU tensor");
    TORCH_CHECK(conv_weight.is_cpu(), "conv_weight must be a CPU tensor");
    TORCH_CHECK(conv_bias.is_cpu(), "conv_bias must be a CPU tensor");
    TORCH_CHECK(bias.is_cpu(), "bias must be a CPU tensor");
    TORCH_CHECK(x.dim() == 4, "x must be of shape (N, C_in, H_in, W_in)");
    TORCH_CHECK(conv_weight.dim() == 4, "conv_weight must be of shape (C_out, C_in, K_h, K_w)");
    TORCH_CHECK(conv_bias.dim() == 1, "conv_bias must be of shape (C_out)");
    TORCH_CHECK(bias.dim() == 3 || bias.dim() == 1,
        "bias must be of shape (C_out, 1, 1) or (C_out,).");

    const auto N     = x.size(0);
    const auto C_in  = x.size(1);
    const auto H_in  = x.size(2);
    const auto W_in  = x.size(3);
    const auto C_out = conv_weight.size(0);
    const auto K_h   = conv_weight.size(2);
    const auto K_w   = conv_weight.size(3);

    auto H_out = H_in - K_h + 1;
    auto W_out = W_in - K_w + 1;
    TORCH_CHECK(H_out > 0 && W_out > 0,
                "Output size (H_out, W_out) must be positive. Check kernel size vs input.");

    x            = x.contiguous();
    conv_weight  = conv_weight.contiguous();
    conv_bias    = conv_bias.contiguous();
    bias         = bias.contiguous();

    auto out = torch::empty({N, C_out, H_out, W_out}, x.options());

    const int total_threads = N * C_out * H_out * W_out;

    #pragma omp parallel for
    for (int idx = 0; idx < total_threads; ++idx) {
        int w_out_idx = idx % W_out;
        int tmp       = idx / W_out;
        int h_out_idx = tmp % H_out;
        tmp           = tmp / H_out;
        int co        = tmp % C_out;
        int n         = tmp / C_out;

        float val = conv_bias[co];

        for (int ci = 0; ci < C_in; ci++) {
            for (int kh = 0; kh < K_h; kh++) {
                for (int kw = 0; kw < K_w; kw++) {
                    int x_h = h_out_idx + kh;
                    int x_w = w_out_idx + kw;
                    float x_val = x[(n * C_in + ci) * H_in * W_in + x_h * W_in + x_w];
                    float w_val = conv_weight[(co * C_in + ci) * K_h * K_w + kh * K_w + kw];
                    val += x_val * w_val;
                }
            }
        }

        val = std::fmaxf(val, 0.0f);
        val += bias[co];
        out[idx] = val;
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &conv2d_relu_bias_forward,
        "Forward pass for 2D convolution + ReLU + bias (CPU)"
    );
}