#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

__global__ void conv2d_relu_bias_kernel(
    const float* __restrict__ x,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int N,
    const int C_in,
    const int H_in,
    const int W_in,
    const int C_out,
    const int K_h,
    const int K_w,
    const int H_out,
    const int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * H_out * W_out) {
        return;
    }

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
                float x_val = x[((n * C_in + ci) * H_in + x_h) * W_in + x_w];
                float w_val = conv_weight[(((co * C_in) + ci) * K_h + kh) * K_w + kw];
                val += x_val * w_val;
            }}
    }

    val = fmaxf(val, 0.0f);
    val += bias[co];
    out[idx] = val;
}

torch::Tensor conv2d_relu_bias_forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor bias)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(conv_weight.is_cuda(), "conv_weight must be a CUDA tensor");
    TORCH_CHECK(conv_bias.is_cuda(), "conv_bias must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
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
    const int blockSize     = 128;  // Changed from 256 to 128
    const int gridSize      = (total_threads + blockSize - 1) / blockSize;

    const float* x_ptr         = x.data_ptr<float>();
    const float* weight_ptr    = conv_weight.data_ptr<float>();
    const float* conv_bias_ptr = conv_bias.data_ptr<float>();
    const float* bias_ptr      = bias.data_ptr<float>();
    float* out_ptr             = out.data_ptr<float>();

    conv2d_relu_bias_kernel<<<gridSize, blockSize>>>(
        x_ptr,
        weight_ptr,
        conv_bias_ptr,
        bias_ptr,
        out_ptr,
        (int)N,
        (int)C_in,
        (int)H_in,
        (int)W_in,
        (int)C_out,
        (int)K_h,
        (int)K_w,
        (int)H_out,
        (int)W_out
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &conv2d_relu_bias_forward,
        "Forward pass for 2D convolution + ReLU + bias (CUDA)"
    );
}