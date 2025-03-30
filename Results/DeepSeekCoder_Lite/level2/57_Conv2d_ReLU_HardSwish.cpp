#include <torch/extension.h>
#include <omp.h>

#define BLOCK_SIZE 128
#define WARP_SIZE 32

__forceinline__ float compute_conv_block(
    const float* __restrict__ x,
    const float* __restrict__ w,
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
    #pragma omp parallel for reduction(+:sum)
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

    const int total_elements = N * C_out * H_out * W_out;

    #pragma omp parallel for
    for (int tid = 0; tid < total_elements; tid++) {
        const int ow = tid % W_out;
        int tmp = tid / W_out;
        const int oh = tmp % H_out;
        tmp /= H_out;
        const int oc = tmp % C_out;
        const int n = tmp / C_out;

        const int xBase = n * C_in * H * W;
        const int wBase = oc * C_in * K * K;

        float val = compute_conv_block(x.data_ptr<float>(), w.data_ptr<float>(), xBase, wBase, C_in, H, W, K, oh, ow);
        
        // Add bias, apply ReLU and HardSwish in one go
        val = std::fmaxf(val + b[oc], 0.0f);
        val *= (val + 3.0f) / 6.0f;

        output[tid] = val;
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Block-tuned Conv2D + ReLU + HardSwish forward (CPU)");
}