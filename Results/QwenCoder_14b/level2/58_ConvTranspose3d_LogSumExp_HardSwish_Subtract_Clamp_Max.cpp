#include <torch/extension.h>
#include <cmath>
#include <omp.h>

void fused_post_ops_cpu(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const float *input,  // conv_transpose3d output: shape [N, C, D, H, W]
    const float *bias,   // bias tensor: shape [N, 1, D, H, W]
    float *output        // output tensor: shape [N, 1, D, H, W]
) {
    #pragma omp parallel for collapse(4)
    for (int n_idx = 0; n_idx < N; ++n_idx) {
        for (int d_idx = 0; d_idx < D; ++d_idx) {
            for (int h_idx = 0; h_idx < H; ++h_idx) {
                for (int w_idx = 0; w_idx < W; ++w_idx) {
                    int base_offset = n_idx * (C * D * H * W) + d_idx * (H * W) + h_idx * W + w_idx;
                    int strideC = D * H * W;

                    // Compute logsumexp over the channel dimension
                    float max_val = -FLT_MAX;
                    for (int c = 0; c < C; ++c) {
                        float val = input[base_offset + c * strideC];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }

                    float sumExp = 0.0f;
                    for (int c = 0; c < C; ++c) {
                        float val = input[base_offset + c * strideC];
                        sumExp += expf(val - max_val);
                    }
                    float lse = max_val + logf(sumExp);

                    // HardSwish activation: x * sigmoid(x + 3) / 6
                    float sigmoid_term = 1.0f / (1.0f + expf(-(lse + 3.0f)));
                    float hswish = lse * sigmoid_term / 6.0f;

                    // Subtract bias
                    int bias_offset = n_idx * (D * H * W) + d_idx * (H * W) + h_idx * W + w_idx;
                    float result = hswish - bias[bias_offset];

                    // Clamp the result to [-1, 1]
                    result = (result < -1.0f) ? -1.0f : result;
                    result = (result > 1.0f) ?  1.0f : result;

                    // Store the result
                    output[n_idx * (D * H * W) + d_idx * (H * W) + h_idx * W + w_idx] = result;
                }
            }
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    torch::Tensor bias
) {
    // Check inputs are CPU tensors
    TORCH_CHECK(!x.is_cuda(), "Input x must be a CPU tensor");
    TORCH_CHECK(!conv_transpose.is_cuda(), "Weights must be a CPU tensor");
    TORCH_CHECK(!conv_transpose_bias.is_cuda(), "Conv-transpose bias must be a CPU tensor");
    TORCH_CHECK(!bias.is_cuda(), "Subtraction bias must be a CPU tensor");

    // 1) 3D transposed convolution
    auto conv_out = at::conv_transpose3d(
        x,
        conv_transpose,
        conv_transpose_bias,
        {stride, stride, stride},
        {padding, padding, padding}
    );

    // conv_out shape: [N, C, D, H, W]
    auto sizes = conv_out.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];

    // Allocate output tensor with shape [N, 1, D, H, W]
    auto output = torch::empty({N, 1, D, H, W}, conv_out.options());

    // Perform the fused operations on the CPU
    fused_post_ops_cpu(
        N, C, D, H, W,
        conv_out.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>()
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused 3D Transposed Conv -> LogSumExp -> HardSwish -> Subtract -> Clamp -> Max (Optimized CPU)");
}