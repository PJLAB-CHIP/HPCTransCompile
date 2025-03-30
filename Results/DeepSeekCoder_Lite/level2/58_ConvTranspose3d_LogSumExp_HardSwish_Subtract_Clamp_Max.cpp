#include <torch/extension.h>
#include <cfloat>
#include <math.h>
#include <omp.h>

// Fused kernel that performs logsumexp (across channels), HardSwish, bias subtraction, clamp,
// and final max reduction (which is trivial after reduction) in a single pass.
// It leverages __ldg() for read-only global memory accesses and assumes input tensors are
// allocated with 128-bit alignment (as is typical with PyTorch CUDA allocations).

void fused_post_ops_kernel(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const float * __restrict__ input,  // conv_transpose3d output: shape [N, C, D, H, W]
    const float * __restrict__ bias,   // bias tensor: shape [N, 1, D, H, W]
    float * __restrict__ output        // output tensor: shape [N, 1, D, H, W]
) {
    #pragma omp parallel for
    for (int idx = 0; idx < N * D * H * W; ++idx) {
        // Compute spatial index (n, d, h, w) from linear index (note: after conv, tensor is [N, C, D, H, W])
        int w_idx = idx % W;
        int temp = idx / W;
        int h_idx = temp % H;
        temp = temp / H;
        int d_idx = temp % D;
        int n_idx = temp / D;

        // In a contiguous tensor with shape [N, C, D, H, W], the layout is:
        // index = n*(C*D*H*W) + c*(D*H*W) + d*(H*W) + h*W + w
        // For a fixed (n, d, h, w) location, each channel is separated by a stride of (D*H*W).
        int strideC = D * H * W;
        int base_offset = n_idx * (C * strideC) + d_idx * (H * W) + h_idx * W + w_idx;

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
        // bias is assumed to be of shape [N, 1, D, H, W]
        int bias_offset = n_idx * (D * H * W) + d_idx * (H * W) + h_idx * W + w_idx;
        float result = hswish - bias[bias_offset];

        // Clamp the result to [-1, 1]
        result = (result < -1.0f) ? -1.0f : result;
        result = (result > 1.0f) ?  1.0f : result;

        // The final max reduction over the channel dimension is trivial after logsumexp, so we directly store the result
        output[idx] = result;
    }
}

// Forward function that first performs the 3D transposed convolution using ATen's conv_transpose3d,
// then fuses the subsequent operations (logsumexp, HardSwish, subtract bias, clamp, and max reduction)
// into a single CPU kernel launch.

torch::Tensor forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    torch::Tensor bias
) {
    // Check inputs are CUDA tensors
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(conv_transpose.is_cuda(), "Weights must be a CUDA tensor");
    TORCH_CHECK(conv_transpose_bias.is_cuda(), "Conv-transpose bias must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Subtraction bias must be a CUDA tensor");

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

    // Launch the fused kernel over all spatial positions (N * D * H * W threads)
    int total_threads = N * D * H * W;

    fused_post_ops_kernel(
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