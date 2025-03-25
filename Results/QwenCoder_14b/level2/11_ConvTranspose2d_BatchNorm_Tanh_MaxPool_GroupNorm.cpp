#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cmath>
#include <omp.h>

void tanh_maxpool_cpu(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int out_H, int out_W) {
    
    #pragma omp parallel for collapse(3)
    for(int n = 0; n < N; n++) {
        for(int c = 0; c < C; c++) {
            for(int out_y = 0; out_y < out_H; out_y++) {
                for(int out_x = 0; out_x < out_W; out_x++) {
                    const int in_y = out_y * 2;
                    const int in_x = out_x * 2;
                    
                    const int base_idx = ((n * C + c) * H + in_y) * W + in_x;
                    
                    float max_val = -INFINITY;
                    for(int dy = 0; dy < 2; dy++) {
                        for(int dx = 0; dx < 2; dx++) {
                            const float val = tanhf(input[base_idx + dy * W + dx]);
                            max_val = fmaxf(max_val, val);
                        }
                    }
                    
                    output[((n * C + c) * out_H + out_y) * out_W + out_x] = max_val;
                }
            }
        }
    }
}

at::Tensor forward(
    at::Tensor x,
    int64_t stride,
    int64_t padding,
    at::Tensor conv_transpose,
    at::Tensor conv_transpose_bias,
    at::Tensor batch_norm_weight,
    at::Tensor batch_norm_bias,
    at::Tensor batch_norm_running_mean,
    at::Tensor batch_norm_running_var,
    at::Tensor group_norm_weight,
    at::Tensor group_norm_bias,
    int64_t num_groups) {
    
    x = at::conv_transpose2d(x, conv_transpose, conv_transpose_bias, {stride, stride}, {padding, padding});
    x = at::batch_norm(x, batch_norm_weight, batch_norm_bias, batch_norm_running_mean,
                      batch_norm_running_var, true, 0.1, 1e-5, true);

    const int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const int out_H = H/2, out_W = W/2;
    at::Tensor y = at::empty({N, C, out_H, out_W}, x.options());

    tanh_maxpool_cpu(x.data_ptr<float>(), y.data_ptr<float>(), N, C, H, W, out_H, out_W);
    
    return at::group_norm(y, num_groups, group_norm_weight, group_norm_bias, 1e-5, true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized fused ConvTranspose2d with reduced warp divergence");
}