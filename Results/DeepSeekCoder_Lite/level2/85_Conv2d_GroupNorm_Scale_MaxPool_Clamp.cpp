#include <torch/extension.h>
#include <ATen/ATen.h>
#include <omp.h>
#include <cmath>
#include <float.h>

// Fused kernel to perform scaling, max pooling and clamping using grid-stride loops
// It processes the output of the prior convolution and group normalization steps.
// For each output element (from pooling), the kernel iterates over the pooling window
// with correct boundary handling, multiplies by a per-channel scale, computes the max,
// and then clamps the result.

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

    // 4) Launch the fused CPU kernel
    int total_output = N * C * outH * outW;
    #pragma omp parallel for
    for (int index = 0; index < total_output; ++index) {
        // Decode flattened index into (n, c, ph, pw)
        int tmp = index;
        int pw = tmp % outW;
        tmp /= outW;
        int ph = tmp % outH;
        tmp /= outH;
        int c = tmp % C;
        tmp /= C;
        int n = tmp;

        // Determine the starting coordinates in the input tensor for this pooling window
        int start_h = ph * maxpool_kernel_size;
        int start_w = pw * maxpool_kernel_size;

        // Initialize max value to lowest possible float
        float max_val = -FLT_MAX;

        // Compute the effective window with boundary checks
        int h_end = start_h + maxpool_kernel_size;
        int w_end = start_w + maxpool_kernel_size;
        if (h_end > H) h_end = H;
        if (w_end > W) w_end = W;

        // Loop over the pooling window
        for (int i_h = start_h; i_h < h_end; i_h++) {
            for (int i_w = start_w; i_w < w_end; i_w++) {
                int input_index = ((n * C + c) * H + i_h) * W + i_w;
                // Multiply by the per-channel scale (broadcasted along H and W)
                float val = gn_out[n][c][i_h][i_w] * scale[c];
                max_val = std::fmax(max_val, val);
            }
        }

        // Apply clamping
        float result = std::fmin(std::fmax(max_val, static_cast<float>(clamp_min)), static_cast<float>(clamp_max));
        z[n][c][ph][pw] = result;
    }

    // Return the final output after max pooling and clamping
    return z;
}

// Pybind11 module definition exposing the forward function
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Custom CPU forward with stride loops");
}