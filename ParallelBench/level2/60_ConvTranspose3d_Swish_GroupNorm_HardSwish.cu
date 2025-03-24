#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ void warp_reduce_double(float& sum, float& sumsq) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
    }
}

template <int WARPS_PER_BLOCK>
__global__ void warp_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int N, int C, int D, int H, int W,
    int groups,
    float eps
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    const int tid = threadIdx.x;
    const int wid = tid / warpSize;
    const int lane = tid % warpSize;
    
    const int n = blockIdx.x;
    const int g = blockIdx.y;
    
    const int channels_per_group = C / groups;
    const int group_size = channels_per_group * D * H * W;
    const int base = n * (C * D * H * W) + g * group_size;
    
    // Per-thread accumulators
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    
    // Process elements with warp-stride loops
    constexpr int VECTOR_SIZE = 4;
    const int warp_offset = wid * warpSize + lane;
    const int stride = WARPS_PER_BLOCK * warpSize;
    
    // Aligned vectorized processing
    const int aligned_size = (group_size / VECTOR_SIZE) * VECTOR_SIZE;
    for (int i = warp_offset * VECTOR_SIZE; i < aligned_size; i += stride * VECTOR_SIZE) {
        float4 vec = *reinterpret_cast<const float4*>(input + base + i);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float x = ((float*)&vec)[j];
            float sw = x / (1.0f + expf(-x));
            local_sum += sw;
            local_sumsq += sw * sw;
        }
    }
    
    // Handle remaining elements
    for (int i = aligned_size + warp_offset; i < group_size; i += stride) {
        float x = __ldg(&input[base + i]);
        float sw = x / (1.0f + expf(-x));
        local_sum += sw;
        local_sumsq += sw * sw;
    }
    
    // Warp-level reduction
    warp_reduce_double(local_sum, local_sumsq);
    
    // First thread in each warp aggregates to global sum using atomics
    __shared__ float mean, inv_std;
    if (lane == 0) {
        atomicAdd(&mean, local_sum);
        atomicAdd(&inv_std, local_sumsq);
    }
    block.sync();
    
    // First thread computes final statistics
    if (tid == 0) {
        mean = mean / group_size;
        float variance = inv_std / group_size - mean * mean;
        inv_std = rsqrtf(variance + eps);
    }
    block.sync();
    
    // Apply normalization and activations using the computed statistics
    for (int i = warp_offset * VECTOR_SIZE; i < aligned_size; i += stride * VECTOR_SIZE) {
        float4 in_vec = *reinterpret_cast<const float4*>(input + base + i);
        float4 out_vec;
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int idx = i + j;
            float x = ((float*)&in_vec)[j];
            float sw = x / (1.0f + expf(-x));
            
            const int c = idx / (D * H * W);
            const int gc = g * channels_per_group + c;
            
            float norm = (sw - mean) * inv_std;
            float y = norm * __ldg(&gamma[gc]) + __ldg(&beta[gc]);
            ((float*)&out_vec)[j] = y * fminf(fmaxf(y + 3.0f, 0.0f), 6.0f) / 6.0f;
        }
        
        *reinterpret_cast<float4*>(output + base + i) = out_vec;
    }
    
    // Handle remaining elements
    for (int i = aligned_size + warp_offset; i < group_size; i += stride) {
        float x = __ldg(&input[base + i]);
        float sw = x / (1.0f + expf(-x));
        
        const int c = i / (D * H * W);
        const int gc = g * channels_per_group + c;
        
        float norm = (sw - mean) * inv_std;
        float y = norm * __ldg(&gamma[gc]) + __ldg(&beta[gc]);
        output[base + i] = y * fminf(fmaxf(y + 3.0f, 0.0f), 6.0f) / 6.0f;
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
    
    dim3 grid(x.size(0), groups);
    
    warp_optimized_kernel<WARPS_PER_BLOCK><<<grid, THREADS_PER_BLOCK>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        group_norm_weight.data_ptr<float>(),
        group_norm_bias.data_ptr<float>(),
        x.size(0), x.size(1), x.size(2), x.size(3), x.size(4),
        groups,
        eps
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized fused kernel");
}