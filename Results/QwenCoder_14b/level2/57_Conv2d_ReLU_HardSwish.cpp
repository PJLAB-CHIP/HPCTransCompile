#include <torch/extension.h>
#include <omp.h>
#include <cmath>

#define BLOCK_SIZE 128

float compute_conv_block(
    const float* x,
    const float* w,
    int xBase,
    int wBase,
    int C_in,
    int H,
    int W,
    int K,
    int oh,
    int ow
) {
    float sum = 0.0f;
    for (int ic = 0; ic < C_in; ic++) {
        int xOffset = xBase + ic * H * W;
        int wOffset = wBase + ic * K * K;
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                sum += x[xOffset + (oh + kh) * W + (ow + kw)] * 
                      w[wOffset + kh * K + kw];
            }
        }
    }
    return sum;
}

void optimized_block_tuned_conv2d_cpu(
    const float* x,
    const float* w,
    const float* b,
    float* out,
    int N,
    int C_in,
    int H,
    int W,
    int C_out,
    int K,
    int H_out,
    int W_out
) {
    #pragma omp parallel for collapse(4)
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < C_out; oc++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    const int xBase = n * C_in * H * W;
                    const int wBase = oc * C_in * K * K;
                    float val = compute_conv_block(x, w, xBase, wBase, C_in, H, W, K, oh, ow);
                    val = std::max(val + b[oc], 0.0f);
                    val *= std::min(std::max((val + 3.0f) / 6.0f, 0.0f), 1.0f);
                    out[n * C_out * H_out * W_out + oc * H_out * W_out + oh * W_out + ow] = val;
                }
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b
) {
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(w.dim() == 4, "w must be 4D");
    TORCH_CHECK(b.dim() == 1, "b must be 1D");

    x = x.contiguous();
    w = w.contiguous();
    b = b.contiguous();

    const int N = x.size(0);
    const int C_in = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int C_out = w.size(0);
    const int K = w.size(2);
    
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    TORCH_CHECK(H_out > 0 && W_out > 0, "Kernel size too large for input");

    auto opts = x.options();
    torch::Tensor output = torch::empty({N, C_out, H_out, W_out}, opts);

    optimized_block_tuned_conv2d_cpu(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H, W, C_out, K, H_out, W_out
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Block-tuned Conv2D + ReLU + HardSwish forward (CPU)");
}