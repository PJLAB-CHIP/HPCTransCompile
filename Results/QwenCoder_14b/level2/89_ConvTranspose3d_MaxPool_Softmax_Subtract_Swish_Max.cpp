#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <omp.h>

namespace py = pybind11;

void balanced_fusion_cpu(
    const float* input,        // pooled output: shape [N, C, D, H, W]
    const float* subtract_tensor, // subtract tensor: shape [C] (broadcast over n, d, h, w)
    float* output,               // final output: shape [N, D, H, W]
    int N, int C, int D, int H, int W) {

    #pragma omp parallel for collapse(4)
    for (int n_idx = 0; n_idx < N; ++n_idx) {
        for (int d_idx = 0; d_idx < D; ++d_idx) {
            for (int h_idx = 0; h_idx < H; ++h_idx) {
                for (int w_idx = 0; w_idx < W; ++w_idx) {
                    int strideC = D * H * W;
                    int base0 = n_idx * C * strideC + d_idx * H * W + h_idx * W + w_idx;

                    // 1. Compute maximum value over channels
                    float max_val = -FLT_MAX;
                    for (int c = 0; c < C; c++) {
                        max_val = std::max(max_val, input[base0 + c * strideC]);
                    }

                    // 2. Compute sum of exponentials for softmax normalization
                    float sum_exp = 0.0f;
                    for (int c = 0; c < C; c++) {
                        sum_exp += expf(input[base0 + c * strideC] - max_val);
                    }

                    // 3. Calculate softmax, subtract, apply swish and find the max value over the channels
                    float final_max = -FLT_MAX;
                    for (int c = 0; c < C; c++) {
                        float sm_val = expf(input[base0 + c * strideC] - max_val) / sum_exp;
                        float y = sm_val - subtract_tensor[c];
                        float swish = y / (1.0f + expf(-y)); // swish activation
                        final_max = std::max(final_max, swish);
                    }

                    // Write to output
                    output[n_idx * D * H * W + d_idx * H * W + h_idx * W + w_idx] = final_max;
                }
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t pool_kernel_size,
    int64_t pool_stride,
    int64_t pool_padding,
    torch::Tensor conv_transpose_weight,
    torch::Tensor conv_transpose_bias,
    torch::Tensor subtract_tensor
) {
    auto conv_out = at::conv_transpose3d(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        {stride, stride, stride},
        {padding, padding, padding},
        {output_padding, output_padding, output_padding},
        1,
        {1, 1, 1}
    );

    auto pool_out = at::max_pool3d(
        conv_out,
        {pool_kernel_size, pool_kernel_size, pool_kernel_size},
        {pool_stride, pool_stride, pool_stride},
        {pool_padding, pool_padding, pool_padding}
    );

    int N = pool_out.size(0);
    int C = pool_out.size(1);
    int D = pool_out.size(2);
    int H = pool_out.size(3);
    int W = pool_out.size(4);

    auto output = at::empty({N, D, H, W}, pool_out.options());

    balanced_fusion_cpu(
        pool_out.data_ptr<float>(),
        subtract_tensor.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced CPU forward pass with optimized workload distribution");
}