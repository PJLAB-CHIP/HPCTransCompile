#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128
#define WARP_SIZE 32

__forceinline__ __device__ float compute_conv_block(
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
    #pragma unroll
    for (int ic = 0; ic < C_in; ic++) {
        int xOffset = xBase + ic * H * W;
        int wOffset = wBase + ic * K * K;
        #pragma unroll
        for (int kh = 0; kh < K; kh++) {
            #pragma unroll
            for (int kw = 0; kw < K; kw++) {
                sum += x[xOffset + (oh + kh) * W + (ow + kw)] * 
                      w[wOffset + kh * K + kw];
            }
        }
    }
    return sum;
}

__global__ void optimized_block_tuned_conv2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int N,
    int C_in,
    int H,
    int W,
    int C_out,
    int K,
    int H_out,
    int W_out
) {
    const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int total_elements = N * C_out * H_out * W_out;
    
    if (tid >= total_elements) return;

    const int ow = tid % W_out;
    int tmp = tid / W_out;
    const int oh = tmp % H_out;
    tmp /= H_out;
    const int oc = tmp % C_out;
    const int n = tmp / C_out;

    const int xBase = n * C_in * H * W;
    const int wBase = oc * C_in * K * K;

    float val = compute_conv_block(x, w, xBase, wBase, C_in, H, W, K, oh, ow);
    
    // Add bias, apply ReLU and HardSwish in one go
    val = fmaxf(val + b[oc], 0.0f);
    val *= __saturatef((val + 3.0f) / 6.0f);

    out[tid] = val;
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
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    optimized_block_tuned_conv2d_kernel<<<num_blocks, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H, W, C_out, K, H_out, W_out
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Block-tuned Conv2D + ReLU + HardSwish forward (CUDA)");
}