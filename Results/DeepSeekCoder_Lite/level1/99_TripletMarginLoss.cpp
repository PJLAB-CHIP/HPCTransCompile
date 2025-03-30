#include <torch/extension.h>
#include <omp.h>
#include <math.h>
#include <vector>

// CPU implementation of the triplet margin loss kernel
void triplet_margin_loss_kernel_warp_shfl(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ output,
    const float margin,
    const int batch_size,
    const int feat_size,
    const int threads) {

    #pragma omp parallel for
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        int offset = batch_idx * feat_size;
        float sum_pos = 0.f;
        float sum_neg = 0.f;

        // Vectorized load using float4 for aligned memory access
        int vectorized_end = (feat_size / 4) * 4;
        for (int i = omp_get_thread_num() * (feat_size / threads) + 0; i < vectorized_end; i += threads) {
            float4 a = {anchor[offset + i], anchor[offset + i + 1], anchor[offset + i + 2], anchor[offset + i + 3]};
            float4 p = {positive[offset + i], positive[offset + i + 1], positive[offset + i + 2], positive[offset + i + 3]};
            float4 n = {negative[offset + i], negative[offset + i + 1], negative[offset + i + 2], negative[offset + i + 3]};
            float d;
            // Accumulate squared differences for positive
            d = a.x - p.x; sum_pos += d * d;
            d = a.y - p.y; sum_pos += d * d;
            d = a.z - p.z; sum_pos += d * d;
            d = a.w - p.w; sum_pos += d * d;
            
            // Accumulate squared differences for negative
            d = a.x - n.x; sum_neg += d * d;
            d = a.y - n.y; sum_neg += d * d;
            d = a.z - n.z; sum_neg += d * d;
            d = a.w - n.w; sum_neg += d * d;
        }

        // Process remaining elements
        for (int i = vectorized_end + omp_get_thread_num(); i < feat_size; i += threads) {
            float a = anchor[offset + i];
            float p = positive[offset + i];
            float n = negative[offset + i];
            float d = a - p;
            sum_pos += d * d;
            d = a - n;
            sum_neg += d * d;
        }

        // Intra-warp reduction using __shfl_down_sync
        unsigned int warp_mask = 0xffffffff;
        sum_pos = __builtin_expect(sum_pos, 0);
        sum_neg = __builtin_expect(sum_neg, 0);
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sum_pos += __builtin_expect(__shfl_down_sync(warp_mask, sum_pos, offset), 0);
            sum_neg += __builtin_expect(__shfl_down_sync(warp_mask, sum_neg, offset), 0);
        }

        // Each warp's lane 0 holds the partial sum
        __shared__ float shared_pos[32];
        __shared__ float shared_neg[32];
        int lane = omp_get_thread_num() % warpSize;
        int warpId = omp_get_thread_num() / warpSize;
        if (lane == 0) {
            shared_pos[warpId] = sum_pos;
            shared_neg[warpId] = sum_neg;
        }
        __builtin_expect(__syncwarp(0), 0);

        // Final reduction: only the first numWarps threads participate
        int numWarps = 256 / warpSize; // assuming blockDim.x is a multiple of warpSize
        if (omp_get_thread_num() < numWarps) {
            float final_sum_pos = shared_pos[omp_get_thread_num() / warpSize];
            float final_sum_neg = shared_neg[omp_get_thread_num() / warpSize];
            // Use warp-level reduction over the participating warp leaders
            for (int off = numWarps / 2; off > 0; off /= 2) {
                final_sum_pos += __builtin_expect(__shfl_down_sync(warp_mask, final_sum_pos, off), 0);
                final_sum_neg += __builtin_expect(__shfl_down_sync(warp_mask, final_sum_neg, off), 0);
            }
            if (omp_get_thread_num() == 0) {
                float loss = sqrtf(final_sum_pos) - sqrtf(final_sum_neg) + margin;
                output[batch_idx] = (loss > 0.f) ? loss : 0.f;
            }
        }
    }
}

// CPU launcher function
torch::Tensor triplet_margin_loss_cpu_optimized(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {

    TORCH_CHECK(anchor.device().is_cpu(), "anchor must be a CPU tensor");
    TORCH_CHECK(positive.device().is_cpu(), "positive must be a CPU tensor");
    TORCH_CHECK(negative.device().is_cpu(), "negative must be a CPU tensor");

    const int batch_size = anchor.size(0);
    const int feat_size = anchor.size(1);
    auto output = torch::empty({batch_size}, anchor.options());

    int threads = omp_get_max_threads();
    triplet_margin_loss_kernel_warp_shfl(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        output.data_ptr<float>(),
        margin,
        batch_size,
        feat_size,
        threads);

    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &triplet_margin_loss_cpu_optimized, "Triplet margin loss forward optimized with warp shfl reduction (CPU)");
}