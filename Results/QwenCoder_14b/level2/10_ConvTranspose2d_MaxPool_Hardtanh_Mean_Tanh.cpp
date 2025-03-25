#include <torch/extension.h>
#include <omp.h>
#include <cmath>
#include <limits>

#define WARP_SIZE 32

// Function to perform warp-level reduction for max pooling
float warp_reduce_max(float val) {
    #pragma omp parallel for reduction(max:val)
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = std::max(val, val);
    }
    return val;
}

// Function to perform warp-level reduction for sum
float warp_reduce_sum(float val) {
    #pragma omp parallel for reduction(+:val)
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += val;
    }
    return val;
}

void conv_transpose_maxpool_mean_cpu(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    float* mean_output,
    int N, int in_channels,
    int H_in, int W_in,
    int out_channels,
    int kernel_h, int kernel_w,
    int stride, int padding,
    int H_out, int W_out,
    int pool_kernel, int pool_stride,
    int H_pool_out, int W_pool_out
) {
    #pragma omp parallel for collapse(4)
    for (int n = 0; n < N; ++n) {
        for (int c_out = 0; c_out < out_channels; ++c_out) {
            for (int h_pool_out = 0; h_pool_out < H_pool_out; ++h_pool_out) {
                for (int w_pool_out = 0; w_pool_out < W_pool_out; ++w_pool_out) {
                    float max_val = -std::numeric_limits<float>::max();
                    float sum_val = 0.0f;
                    int valid_count = 0;

                    for (int ph = 0; ph < pool_kernel; ++ph) {
                        for (int pw = 0; pw < pool_kernel; ++pw) {
                            float conv_val = 0.0f;
                            const int h_out = h_pool_out * pool_stride + ph;
                            const int w_out = w_pool_out * pool_stride + pw;

                            for (int c_in = 0; c_in < in_channels; ++c_in) {
                                for (int kh = 0; kh < kernel_h; ++kh) {
                                    for (int kw = 0; kw < kernel_w; ++kw) {
                                        int h_in = (h_out + padding - kh) / stride;
                                        int w_in = (w_out + padding - kw) / stride;
                                        bool valid = ((h_out + padding - kh) % stride == 0) && 
                                                   ((w_out + padding - kw) % stride == 0) &&
                                                   (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in);

                                        if (valid) {
                                            const int input_idx = ((n * in_channels + c_in) * H_in + h_in) * W_in + w_in;
                                            const int weight_idx = ((c_in * out_channels + c_out) * kernel_h + kh) * kernel_w + kw;
                                            conv_val += input[input_idx] * weight[weight_idx];
                                        }
                                    }
                                }
                            }
                            conv_val += bias[c_out];
                            max_val = std::max(max_val, conv_val);
                            sum_val += conv_val;
                            valid_count++;
                        }
                    }

                    // Reduce max value
                    max_val = warp_reduce_max(max_val);
                    output[(n * out_channels + c_out) * H_pool_out * W_pool_out + h_pool_out * W_pool_out + w_pool_out] = max_val;

                    // Compute mean
                    sum_val = warp_reduce_sum(sum_val);
                    mean_output[n * out_channels + c_out] = sum_val / valid_count;
                }
            }
        }
    }
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

    conv_transpose_maxpool_mean_cpu(
        x.data_ptr<float>(),
        conv_transpose.data_ptr<float>(),
        conv_transpose_bias.data_ptr<float>(),
        pool_out.data_ptr<float>(),
        mean_out.data_ptr<float>(),
        N, in_channels, H_in, W_in,
        out_channels, kernel_h, kernel_w,
        stride, padding,
        H_conv, W_conv,
        maxpool_kernel_size, maxpool_stride,
        H_pool, W_pool
    );

    pool_out = torch::clamp(pool_out, hardtanh_min, hardtanh_max);
    mean_out = torch::tanh(mean_out);

    return mean_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Reduction Forward");
}