#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <omp.h>
#include <cmath>
#include <limits>

// Fused CPU function to perform scaling, max pooling and clamping
void fused_scale_maxpool_clamp_cpu(
    const float* input,   // Input tensor (N, C, H, W)
    float* output,        // Output tensor (N, C, outH, outW)
    const float* scale,   // Per-channel scale vector (C), broadcasted
    int N, int C, int H, int W,          // Dimensions of the input tensor
    int poolKernel,                    // Pooling kernel size
    float clamp_min, float clamp_max,  // Clamping bounds
    int outH, int outW                 // Dimensions of the output tensor
) {
    #pragma omp parallel for collapse(4)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int ph = 0; ph < outH; ++ph) {
                for (int pw = 0; pw < outW; ++pw) {
                    // Determine the starting coordinates in the input tensor for this pooling window
                    int start_h = ph * poolKernel;
                    int start_w = pw * poolKernel;

                    // Initialize max value to lowest possible float
                    float max_val = -std::numeric_limits<float>::max();

                    // Compute the effective window with boundary checks
                    int h_end = start_h + poolKernel;
                    int w_end = start_w + poolKernel;
                    if (h_end > H) h_end = H;
                    if (w_end > W) w_end = W;

                    // Loop over the pooling window
                    for (int i_h = start_h; i_h < h_end; ++i_h) {
                        for (int i_w = start_w; i_w < w_end; ++i_w) {
                            int input_index = ((n * C + c) * H + i_h) * W + i_w;
                            // Multiply by the per-channel scale (broadcasted along H and W)
                            float val = input[input_index] * scale[c];
                            max_val = std::max(max_val, val);
                        }
                    }

                    // Apply clamping
                    float result = std::min(std::max(max_val, clamp_min), clamp_max);
                    output[((n * C + c) * outH + ph) * outW + pw] = result;
                }
            }
        }
    }
}

// The forward function performs convolution and group normalization using existing ATen operators,
// then fuses scaling, max pooling and clamping into a custom CPU function using OpenMP.

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias,
    torch::Tensor scale,
    int64_t num_groups,
    int64_t maxpool_kernel_size,
    double clamp_min,
    double clamp_max
) {
    // 1) Convolution using ATen operator
    auto conv_out = at::conv2d(x, conv_weight, conv_bias);

    // 2) Group normalization (using eps = 1e-5 and cudnn enabled)
    auto gn_out = at::group_norm(conv_out, num_groups, group_norm_weight, group_norm_bias, 1e-5, true);

    // Get dimensions from the group norm output. Expected layout is [N, C, H, W].
    int N = gn_out.size(0);
    int C = gn_out.size(1);
    int H = gn_out.size(2);
    int W = gn_out.size(3);

    // 3) Allocate output tensor for the fused max pool result.
    // PyTorch's max_pool2d (with stride equal to kernel size) computes output dims as:
    // out_dim = floor((in_dim - kernel_size) / kernel_size) + 1
    int outH = (H - maxpool_kernel_size) / maxpool_kernel_size + 1;
    int outW = (W - maxpool_kernel_size) / maxpool_kernel_size + 1;
    auto z = at::empty({N, C, outH, outW}, gn_out.options());

    // 4) Call the fused CPU function
    fused_scale_maxpool_clamp_cpu(
        gn_out.data_ptr<float>(),
        z.data_ptr<float>(),
        scale.data_ptr<float>(),
        N, C, H, W,
        maxpool_kernel_size,
        static_cast<float>(clamp_min), static_cast<float>(clamp_max),
        outH, outW
    );

    // Return the final output after max pooling and clamping
    return z;
}

// Pybind11 module definition exposing the forward function
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Custom CPU forward with OpenMP");
}