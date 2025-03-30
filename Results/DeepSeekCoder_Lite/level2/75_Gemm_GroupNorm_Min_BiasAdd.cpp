#include <torch/extension.h>
#include <omp.h>
#include <cmath>
#include <float.h>

namespace F = torch::nn::functional;

// Warp-level reduction for minimum using __shfl_down_sync
__forceinline__ float warpReduceMin(float val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val = std::fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Fused kernel: Computes Group Normalization (with manual group stat computation) and min reduction in one pass
void fused_groupnorm_min_kernel(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    const float* bias,
    int batch_size,
    int channels,
    int num_groups,
    int channels_per_group) {

    #pragma omp parallel for
    for (int bid = 0; bid < batch_size; ++bid) {
        const float* row_start = input + bid * channels;

        // Step 1: Compute group statistics: mean and variance (std. deviation) for each group
        float* mean = new float[num_groups];
        float* var = new float[num_groups];
        #pragma omp parallel for
        for (int g = 0; g < num_groups; ++g) {
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
            var[g] = std::sqrt(variance + 1e-5f);
        }

        // Step 2: Fused normalization, transformation and min reduction
        float thread_min = FLT_MAX;
        #pragma omp parallel for
        for (int c = 0; c < channels; ++c) {
            int group = c / channels_per_group;
            float norm = (row_start[c] - mean[group]) / var[group];
            float transformed = gamma[c] * norm + beta[c];
            thread_min = std::fminf(thread_min, transformed);
        }

        // Warp-level reduction using __shfl_down_sync
        thread_min = warpReduceMin(thread_min);
        int lane = 0; // Assuming threadIdx.x is used for warp identification
        int warp_id = 0; // Assuming threadIdx.x / warpSize is used for warp identification

        // Use shared memory to collect each warp's minimum
        __shared__ float warp_min[32]; // supports up to 32 warps per block
        if (lane == 0) {
            warp_min[warp_id] = thread_min;
        }
        __syncthreads();

        // Final reduction within the first warp
        float block_min = FLT_MAX;
        if (0 < (32 / warpSize)) {
            block_min = warp_min[lane];
            block_min = warpReduceMin(block_min);
        }

        // Thread 0 writes the final minimum value with bias added
        output[bid] = block_min + bias[bid];

        delete[] mean;
        delete[] var;
    }
}

// Forward function: Fuses GEMM, Group Normalization and min reduction
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

    int batch_size = x.size(0);
    int channels = x.size(1);
    int channels_per_group = channels / num_groups;

    auto output = torch::empty({batch_size}, x.options());

    // Launch kernel: each block processes one sample
    int threads_per_block = 256;
    int num_blocks = batch_size;
    int shared_mem_size = 2 * num_groups * sizeof(float); // For mean and variance arrays

    fused_groupnorm_min_kernel(
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