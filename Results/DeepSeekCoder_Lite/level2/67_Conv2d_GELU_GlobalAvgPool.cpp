#include <torch/extension.h>
#include <cmath>
#include <omp.h>

#define KERNEL_SIZE 3
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

__forceinline__ float gelu_activate(float x) {
    return 0.5f * x * (1.f + std::erff(x / 1.41421356f));
}

__forceinline__ void warp_reduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

torch::Tensor forward(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias
) {
    TORCH_CHECK(input.is_cuda() == false, "input must be a CPU tensor");
    TORCH_CHECK(conv_weight.is_cuda() == false, "conv_weight must be a CPU tensor");
    TORCH_CHECK(conv_bias.is_cuda() == false, "conv_bias must be a CPU tensor");

    const int N = input.size(0);
    const int in_channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_channels = conv_weight.size(0);
    const int out_h = in_h - 2;
    const int out_w = in_w - 2;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::empty({N, out_channels}, options);

    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        for (int c_out = 0; c_out < out_channels; ++c_out) {
            float* output_ptr = output.data_ptr<float>() + n * out_channels + c_out;
            const float* input_ptr = input.data_ptr<float>() + (n * in_channels + c_out) * in_h * in_w;
            const float* conv_weight_ptr = conv_weight.data_ptr<float>() + c_out * in_channels * KERNEL_SIZE * KERNEL_SIZE;
            const float* conv_bias_ptr = conv_bias.data_ptr<float>() + c_out;

            int total_pixels = out_h * out_w;
            float thread_sum = 0.0f;

            for (int pixel_idx = 0; pixel_idx < total_pixels; ++pixel_idx) {
                int out_row = pixel_idx / out_w;
                int out_col = pixel_idx % out_w;

                float conv_result = 0.0f;

                for (int ic = 0; ic < in_channels; ++ic) {
                    const float* in_ptr = input_ptr + (n * in_channels + ic) * in_h * in_w + out_row * in_w + out_col;
                    const float* w_ptr = conv_weight_ptr + ic * KERNEL_SIZE * KERNEL_SIZE;

                    for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
                        for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                            conv_result += in_ptr[kh * in_w + kw] * w_ptr[kh * KERNEL_SIZE + kw];
                        }
                    }
                }

                conv_result += conv_bias_ptr[0];
                conv_result = gelu_activate(conv_result);
                thread_sum += conv_result;
            }

            thread_sum /= total_pixels;
            *output_ptr = thread_sum;
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized reduction Conv2d + GELU + GlobalAvgPool");
}