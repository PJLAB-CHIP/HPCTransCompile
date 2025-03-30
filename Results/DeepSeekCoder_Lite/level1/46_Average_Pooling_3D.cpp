#include <torch/extension.h>
#include <omp.h>

// Optimized 3D Average Pooling Kernel using combined ideas from two versions:
// - Grid mapping using blockIdx.z to combine (n, c, d_out)
// - Thread block configured as (32, 8, 1) for improved memory coalescing along the width dimension
// - Pointer arithmetic precomputations for efficient inner loop over the pooling window

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    int batch_size = input.size(0);
    int channels   = input.size(1);
    int in_d       = input.size(2);
    int in_h       = input.size(3);
    int in_w       = input.size(4);

    // Calculate output dimensions based on convolution arithmetic
    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    // Configure thread block and grid dimensions for optimal memory access
    int threads = 32;  // 32 threads in width for coalesced global memory accesses
    int blocks = (out_w + threads - 1) / threads * (out_h + 8 - 1) / 8 * batch_size * channels * out_d;

    #pragma omp parallel for
    for (int idx = 0; idx < batch_size * channels * out_d; ++idx) {
        int d_out = idx % out_d;
        idx /= out_d;
        int c = idx % channels;
        idx /= channels;
        int n = idx;

        int h_out = (d_out * stride - padding + omp_get_thread_num()) / out_d;
        int w_out = (c * stride - padding + omp_get_thread_num()) / out_h;
        if (h_out >= out_h || w_out >= out_w) continue;

        int d_start = d_out * stride - padding;
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;

        int d_start_clamped = std::max(d_start, 0);
        int h_start_clamped = std::max(h_start, 0);
        int w_start_clamped = std::max(w_start, 0);
        int d_end_clamped = std::min(d_start + kernel_size, in_d);
        int h_end_clamped = std::min(h_start + kernel_size, in_h);
        int w_end_clamped = std::min(w_start + kernel_size, in_w);

        float sum = 0.0f;
        int pool_volume = kernel_size * kernel_size * kernel_size;
        int baseOffset = (n * channels + c) * in_d;

        for (int d = d_start_clamped; d < d_end_clamped; d++) {
            int d_offset = (baseOffset + d) * in_h * in_w;
            for (int h = h_start_clamped; h < h_end_clamped; h++) {
                int row_start = d_offset + h * in_w + w_start_clamped;
                int row_length = w_end_clamped - w_start_clamped;
                for (int offset = 0; offset < row_length; offset++) {
                    sum += input[n * channels * in_d * in_h * in_w + c * in_d * in_h * in_w + d * in_h * in_w + h * in_w + offset];
                }
            }
        }

        int output_idx = (((n * channels + c) * out_d + d_out) * out_h + h_out) * out_w + w_out;
        output[output_idx] = sum / static_cast<float>(pool_volume);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 3D Average Pooling forward (CPU)");
}