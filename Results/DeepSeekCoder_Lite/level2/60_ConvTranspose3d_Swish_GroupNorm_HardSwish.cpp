#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <omp.h>

namespace cg = cooperative_groups;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<typename T>
__forceinline__ T warp_reduce_sum(T val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__forceinline__ void warp_reduce_double(float& sum, float& sumsq) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
    }
}

torch::Tensor forward(
    torch::Tensor x,
    int stride,
    int padding,
    int groups,
    float eps,
    torch::Tensor conv_transpose,
    torch::Tensor conv_transpose_bias,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias
) {
    CHECK_INPUT(x);
    CHECK_INPUT(conv_transpose);
    CHECK_INPUT(conv_transpose_bias);
    CHECK_INPUT(group_norm_weight);
    CHECK_INPUT(group_norm_bias);

    x = torch::conv_transpose3d(x, conv_transpose, conv_transpose_bias, stride, padding);
    torch::Tensor output = torch::empty_like(x);

    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

    #pragma omp parallel for
    for (int n = 0; n < x.size(0); ++n) {
        for (int g = 0; g < groups; ++g) {
            const int C = x.size(1);
            const int D = x.size(2);
            const int H = x.size(3);
            const int W = x.size(4);
            const int channels_per_group = C / groups;
            const int group_size = channels_per_group * D * H * W;
            const int base = n * (C * D * H * W) + g * group_size;

            // Per-thread accumulators
            float local_sum = 0.0f;
            float local_sumsq = 0.0f;

            // Process elements with warp-stride loops
            constexpr int VECTOR_SIZE = 4;
            const int stride = WARPS_PER_BLOCK * 32;
            const int aligned_size = (group_size / VECTOR_SIZE) * VECTOR_SIZE;
            for (int i = 0; i < aligned_size; i += VECTOR_SIZE) {
                float4 vec = *reinterpret_cast<const float4*>(&x[base + i]);
                for (int j = 0; j < 4; ++j) {
                    float x_val = ((float*)&vec)[j];
                    float sw = x_val / (1.0f + expf(-x_val));
                    local_sum += sw;
                    local_sumsq += sw * sw;
                }
            }
            for (int i = aligned_size; i < group_size; ++i) {
                float x_val = x[base + i];
                float sw = x_val / (1.0f + expf(-x_val));
                local_sum += sw;
                local_sumsq += sw * sw;
            }

            // Warp-level reduction
            warp_reduce_double(local_sum, local_sumsq);

            // First thread in each warp aggregates to global sum using atomics
            __shared__ float mean, inv_std;
            if (threadIdx.x == 0) {
                atomicAdd(&mean, local_sum);
                atomicAdd(&inv_std, local_sumsq);
            }
            __syncthreads();

            // First thread computes final statistics
            if (threadIdx.x == 0) {
                mean = mean / group_size;
                float variance = inv_std / group_size - mean * mean;
                inv_std = 1.0f / sqrtf(variance + eps);
            }
            __syncthreads();

            // Apply normalization and activations using the computed statistics
            const int warp_size = 32;
            for (int i = threadIdx.x; i < group_size; i += warp_size) {
                float x_val = x[base + i];
                float sw = x_val / (1.0f + expf(-x_val));
                float norm = (sw - mean) * inv_std;
                output[base + i] = norm * group_norm_weight[g * channels_per_group + (i / (D * H * W))] + group_norm_bias[g * channels_per_group + (i / (D * H * W))];
            }
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized fused kernel");
}