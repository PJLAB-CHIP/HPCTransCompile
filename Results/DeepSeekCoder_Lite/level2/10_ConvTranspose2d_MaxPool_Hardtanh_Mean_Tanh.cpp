#include <torch/extension.h>
#include <vector>
#include <cfloat>
#include <omp.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

__forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val = std::max(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

__forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    int64_t maxpool_kernel_size,
    int64_t maxpool_stride,
    double hardtanh_min,
    double hardtanh_max,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias
) {
    const int N = x.size(0);
    const int in_channels = x.size(1);
    const int H_in = x.size(2);
    const int W_in = x.size(3);
    const int out_channels = conv_transpose.size(1);
    const int kernel_h = conv_transpose.size(2);
    const int kernel_w = conv_transpose.size(3);

    const int H_conv = (H_in - 1) * stride - 2 * padding + kernel_h;
    const int W_conv = (W_in - 1) * stride - 2 * padding + kernel_w;
    const int H_pool = (H_conv - maxpool_kernel_size) / maxpool_stride + 1;
    const int W_pool = (W_conv - maxpool_kernel_size) / maxpool_stride + 1;

    auto pool_out = torch::empty({N, out_channels, H_pool, W_pool}, x.options());
    auto mean_out = torch::empty({N, out_channels, 1, 1}, x.options());

    const int threads = 256;
    const int total = N * out_channels * H_pool * W_pool;
    const int blocks = (total + threads - 1) / threads;

    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        for (int c_out = 0; c_out < out_channels; ++c_out) {
            for (int h_pool_out = 0; h_pool_out < H_pool; ++h_pool_out) {
                for (int w_pool_out = 0; w_pool_out < W_pool; ++w_pool_out) {
                    float max_val = -FLT_MAX;
                    float sum_val = 0.0f;
                    int valid_count = 0;

                    for (int ph = 0; ph < maxpool_kernel_size; ++ph) {
                        for (int pw = 0; pw < maxpool_kernel_size; ++pw) {
                            int h_out = h_pool_out * maxpool_stride + ph;
                            int w_out = w_pool_out * maxpool_stride + pw;

                            for (int c_in = 0; c_in < in_channels; ++c_in) {
                                for (int kh = 0; kh < kernel_h; ++kh) {
                                    for (int kw = 0; kw < kernel_w; ++kw) {
                                        int h_in = (h_out + padding - kh) / stride;
                                        int w_in = (w_out + padding - kw) / stride;
                                        bool valid = ((h_out + padding - kh) % stride == 0) && 
                                                     ((w_out + padding - kw) % stride == 0) &&
                                                     (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in);

                                        if (valid) {
                                            int input_idx = ((n * in_channels + c_in) * H_in + h_in) * W_in + w_in;
                                            int weight_idx = ((c_in * out_channels + c_out) * kernel_h + kh) * kernel_w + kw;
                                            sum_val += x[input_idx].item<float>() * conv_transpose[weight_idx].item<float>();
                                        }
                                    }
                                }
                            }
                            sum_val += conv_transpose_bias[c_out].item<float>();
                            max_val = std::max(max_val, sum_val);
                            sum_val = 0.0f;
                            valid_count++;
                        }
                    }

                    // Warp-level reduction for max pooling
                    max_val = warp_reduce_max(max_val);
                    pool_out[n][c_out][h_pool_out][w_pool_out] = max_val;

                    // Compute mean using warp-level reduction
                    float mean_val = warp_reduce_sum(sum_val) / valid_count;
                    mean_out[n][c_out][0][0] = mean_val;
                }
            }
        }
    }

    pool_out = torch::clamp(pool_out, hardtanh_min, hardtanh_max);
    mean_out = torch::tanh(mean_out);

    return mean_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Reduction Forward");
}