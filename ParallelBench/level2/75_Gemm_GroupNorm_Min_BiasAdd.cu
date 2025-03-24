#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cmath>

namespace F = torch::nn::functional;

// Warp-level reduction for minimum using __shfl_down_sync
__device__ __forceinline__ float warpReduceMin(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Fused kernel: Computes Group Normalization (with manual group stat computation) and min reduction in one pass
__global__ void fused_groupnorm_min_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ bias,
    const int batch_size,
    const int channels,
    const int num_groups,
    const int channels_per_group) {

    // Dynamically allocated shared memory: first num_groups floats for means, next num_groups for std deviations
    extern __shared__ float shared_mem[];
    float* mean = shared_mem;                 // size: num_groups
    float* var  = shared_mem + num_groups;      // size: num_groups

    int tid = threadIdx.x;
    int bid = blockIdx.x; // each block processes one sample (row)

    if (bid < batch_size) {
        const float* row_start = input + bid * channels;

        // Step 1: Compute group statistics: mean and variance (std. deviation) for each group
        for (int g = tid; g < num_groups; g += blockDim.x) {
            float sum = 0.0f, sum_sq = 0.0f;
            int start = g * channels_per_group;
            int end = start + channels_per_group;
            for (int c = start; c < end; ++c) {
                float v = row_start[c];
                sum += v;
                sum_sq += v * v;
            }
            mean[g] = sum / channels_per_group;
            float variance = sum_sq / channels_per_group - mean[g] * mean[g];
            var[g] = sqrtf(variance + 1e-5f);
        }
        __syncthreads();

        // Step 2: Fused normalization, transformation and min reduction
        float thread_min = FLT_MAX;
        // Each thread processes a strided subset of channels
        for (int c = tid; c < channels; c += blockDim.x) {
            int group = c / channels_per_group;
            float norm = (row_start[c] - mean[group]) / var[group];
            float transformed = gamma[c] * norm + beta[c];
            thread_min = fminf(thread_min, transformed);
        }

        // Warp-level reduction using __shfl_down_sync
        thread_min = warpReduceMin(thread_min);
        int lane = tid % warpSize;
        int warp_id = tid / warpSize;

        // Use shared memory to collect each warp's minimum
        __shared__ float warp_min[32]; // supports up to 32 warps per block
        if (lane == 0) {
            warp_min[warp_id] = thread_min;
        }
        __syncthreads();

        // Final reduction within the first warp
        float block_min = FLT_MAX;
        if (tid < (blockDim.x / warpSize)) {
            block_min = warp_min[lane];
            block_min = warpReduceMin(block_min);
        }

        // Thread 0 writes the final minimum value with bias added
        if (tid == 0) {
            output[bid] = block_min + bias[bid];
        }
    }
}

// Forward function: Fuses GEMM, Group Normalization and min reduction
// The bias addition is fused into the min reduction to eliminate an extra pass

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor gemm_weight,
    torch::Tensor gemm_bias,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias,
    int64_t num_groups,
    torch::Tensor bias) {

    // Ensure all inputs are CUDA tensors
    if (!x.is_cuda() || !gemm_weight.is_cuda() || !gemm_bias.is_cuda() ||
        !group_norm_weight.is_cuda() || !group_norm_bias.is_cuda() || !bias.is_cuda()) {
        throw std::invalid_argument("All inputs must be CUDA tensors");
    }

    // GEMM: perform linear transformation
    x = F::linear(x, gemm_weight, gemm_bias);

    const int batch_size = x.size(0);
    const int channels = x.size(1);
    const int channels_per_group = channels / num_groups;

    auto output = torch::empty({batch_size}, x.options());

    // Launch kernel: each block processes one sample
    const int threads_per_block = 256;
    const int num_blocks = batch_size;
    const int shared_mem_size = 2 * num_groups * sizeof(float); // For mean and variance arrays

    fused_groupnorm_min_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        group_norm_weight.data_ptr<float>(),
        group_norm_bias.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        channels,
        num_groups,
        channels_per_group
    );

    return output.unsqueeze(1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused GroupNorm and Min Reduction with GEMM");
}
