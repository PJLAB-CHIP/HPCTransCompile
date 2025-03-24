#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#define UNROLL_NUM 4

__global__ void unrolled_fused_kernel_singlepass(
    const float* __restrict__ conv,   // Output of conv2d: [N, C, H, W]
    const float* __restrict__ norm,   // Output of group_norm: [N, C, H, W]
    float* __restrict__ out,          // Output: logsumexp over channels: [N, 1, H, W] stored as [N, H*W]
    int C, int H, int W) {

    int n = blockIdx.x;
    int num_pixels = H * W;

    int pixel = blockIdx.y * blockDim.x + threadIdx.x;
    if (pixel >= num_pixels) return;

    int image_offset = n * C * num_pixels;

    float max_val = -FLT_MAX;
    float sum_exp = 0.0f;

    // Manual loop unrolling to enhance performance
    for (int c = 0; c <= C - UNROLL_NUM; c += UNROLL_NUM) {
        #pragma unroll
        for (int i = 0; i < UNROLL_NUM; ++i) {
            int idx = image_offset + (c + i) * num_pixels + pixel;
            float conv_val = conv[idx];
            float norm_val = norm[idx];
            float tanh_val = tanhf(norm_val);
            float hardswish_val = tanh_val * fminf(fmaxf(tanh_val + 3.0f, 0.0f), 6.0f) / 6.0f;
            float value = conv_val + hardswish_val;

            // Compute and update max_val and sum_exp in a single pass
            if (value > max_val) {
                sum_exp = sum_exp * expf(max_val - value) + 1.0f;
                max_val = value;
            } else {
                sum_exp += expf(value - max_val);
            }
        }
    }

    // Handle remaining channels
    for (int c = (C / UNROLL_NUM) * UNROLL_NUM; c < C; ++c) {
        int idx = image_offset + c * num_pixels + pixel;
        float conv_val = conv[idx];
        float norm_val = norm[idx];
        float tanh_val = tanhf(norm_val);
        float hardswish_val = tanh_val * fminf(fmaxf(tanh_val + 3.0f, 0.0f), 6.0f) / 6.0f;
        float value = conv_val + hardswish_val;
        if (value > max_val) {
            sum_exp = sum_exp * expf(max_val - value) + 1.0f;
            max_val = value;
        } else {
            sum_exp += expf(value - max_val);
        }
    }

    int out_idx = n * num_pixels + pixel;
    out[out_idx] = logf(sum_exp) + max_val;
}

torch::Tensor module_fn_forward(
    torch::Tensor x,
    double eps,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias,
    int64_t groups) {

    x = x.contiguous();
    conv_weight = conv_weight.contiguous();
    conv_bias = conv_bias.contiguous();
    group_norm_weight = group_norm_weight.contiguous();
    group_norm_bias = group_norm_bias.contiguous();

    torch::Tensor x_conv = torch::conv2d(x, conv_weight, conv_bias);
    torch::Tensor x_norm = torch::group_norm(x_conv, groups, group_norm_weight, group_norm_bias, eps);

    int N = x_conv.size(0);
    int C = x_conv.size(1);
    int H = x_conv.size(2);
    int W = x_conv.size(3);
    int num_pixels = H * W;

    torch::Tensor out = torch::empty({N, 1, H, W}, x_conv.options());

    dim3 grid(N, (num_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    unrolled_fused_kernel_singlepass<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x_conv.data_ptr<float>(),
        x_norm.data_ptr<float>(),
        out.data_ptr<float>(),
        C, H, W);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Unrolled and fused single-pass kernel with loop unrolling");
}