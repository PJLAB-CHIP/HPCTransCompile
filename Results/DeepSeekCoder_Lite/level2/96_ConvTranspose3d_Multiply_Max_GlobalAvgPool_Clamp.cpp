#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cfloat>
#include <omp.h>

// Fused kernel with stride loops to handle workloads larger than the number of available threads
// Each block is responsible for one (batch, channel) pair.
// The pooling window size is templated as POOL_K for compile-time optimizations.

template<int POOL_K>
void fused_pooling_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N, const int C, const int D, const int H, const int W,
    const float scale, const int n, const int c) {

    const int D_pool = D / POOL_K;
    const int H_pool = H / POOL_K;
    const int W_pool = W / POOL_K;
    const int total_windows = D_pool * H_pool * W_pool;

    const int channel_offset = ((n * C + c) * D * H * W);
    int tid = omp_get_thread_num();
    int stride = omp_get_num_threads();

    float local_sum = 0.0f;

    // Loop over pooling windows in a strided fashion.
    for (int win_idx = tid; win_idx < total_windows; win_idx += stride) {
        const int d_pool_idx = win_idx / (H_pool * W_pool);
        const int rem = win_idx % (H_pool * W_pool);
        const int h_pool_idx = rem / W_pool;
        const int w_pool_idx = rem % W_pool;

        const int d_start = d_pool_idx * POOL_K;
        const int h_start = h_pool_idx * POOL_K;
        const int w_start = w_pool_idx * POOL_K;

        float max_val = -FLT_MAX;

        // Stride loops for pooling window
        for (int i = d_start; i < d_start + POOL_K; i++) {
            for (int j = h_start; j < h_start + POOL_K; j++) {
                for (int k = w_start; k < w_start + POOL_K; k++) {
                    const int index = channel_offset + (i * H * W) + (j * W) + k;
                    const float val = input[index] * scale;
                    max_val = std::max(max_val, val);
                }
            }
        }
        local_sum += max_val;
    }

    // Warp reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    __shared__ float shared_sum[32];
    int lane = tid % 32;
    int warpId = tid / 32;

    if (lane == 0) {
        shared_sum[warpId] = local_sum;
    }
    __syncthreads();

    if (tid < 32) {
        float warp_sum = (tid < ((32 + 32 - 1) / 32)) ? shared_sum[tid] : 0.0f;
        
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }

        if (tid == 0) {
            float avg = warp_sum / total_windows;
            avg = __saturatef(avg); // Clamps to [0,1] range
            output[n * C + c] = avg;
        }
    }
}

torch::Tensor forward(
    torch::Tensor x,
    int64_t stride,
    int64_t padding,
    double scale,
    int64_t maxpool_kernel_size,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias) {

    auto conv_out = torch::conv_transpose3d(
        x, conv_transpose, conv_transpose_bias,
        {stride, stride, stride},
        {padding, padding, padding}
    );

    conv_out = conv_out.contiguous();
    const int N = conv_out.size(0);
    const int C = conv_out.size(1);
    const int D = conv_out.size(2);
    const int H = conv_out.size(3);
    const int W = conv_out.size(4);

    auto output = torch::empty({N, C}, conv_out.options());

    const int threads = 256;
    int num_blocks = (C * N + threads - 1) / threads;

    // Template specialization based on pool size
    if (maxpool_kernel_size == 2) {
        #pragma omp parallel for
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                fused_pooling_kernel<2>(
                    conv_out.data_ptr<float>(), output.data_ptr<float>(),
                    N, C, D, H, W, static_cast<float>(scale), n, c);
            }
        }
    } else {
        #pragma omp parallel for
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                fused_pooling_kernel<4>(
                    conv_out.data_ptr<float>(), output.data_ptr<float>(),
                    N, C, D, H, W, static_cast<float>(scale), n, c);
            }
        }
    }

    return output.view({N, C, 1, 1, 1});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp CPU kernel with stride loops");
}