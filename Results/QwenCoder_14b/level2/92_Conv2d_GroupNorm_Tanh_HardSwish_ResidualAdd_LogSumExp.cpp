#include <torch/extension.h>
#include <cmath>
#include <omp.h>

#define UNROLL_NUM 4

void unrolled_fused_cpu_singlepass(
    const float* conv,   // Output of conv2d: [N, C, H, W]
    const float* norm,   // Output of group_norm: [N, C, H, W]
    float* out,          // Output: logsumexp over channels: [N, 1, H, W] stored as [N, H*W]
    int N, int C, int H, int W) {

    int num_pixels = H * W;

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int pixel = 0; pixel < num_pixels; ++pixel) {
            int image_offset = n * C * num_pixels;

            float max_val = -INFINITY;
            float sum_exp = 0.0f;

            // Manual loop unrolling to enhance performance
            for (int c = 0; c <= C - UNROLL_NUM; c += UNROLL_NUM) {
                #pragma unroll
                for (int i = 0; i < UNROLL_NUM; ++i) {
                    int idx = image_offset + (c + i) * num_pixels + pixel;
                    float conv_val = conv[idx];
                    float norm_val = norm[idx];
                    float tanh_val = tanhf(norm_val);
                    float hardswish_val = tanh_val * std::fmin(std::fmax(tanh_val + 3.0f, 0.0f), 6.0f) / 6.0f;
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
                float hardswish_val = tanh_val * std::fmin(std::fmax(tanh_val + 3.0f, 0.0f), 6.0f) / 6.0f;
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
    }
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

    torch::Tensor out = torch::empty({N, 1, H, W}, x_conv.options());

    unrolled_fused_cpu_singlepass(
        x_conv.data_ptr<float>(),
        x_norm.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Unrolled and fused single-pass kernel with loop unrolling");
}