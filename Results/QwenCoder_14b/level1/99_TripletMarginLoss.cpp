#include <torch/extension.h>
#include <cmath>
#include <omp.h>

void triplet_margin_loss_cpu(
    const float* anchor,
    const float* positive,
    const float* negative,
    float* output,
    const float margin,
    const int batch_size,
    const int feat_size) {

    #pragma omp parallel for
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        int offset = batch_idx * feat_size;
        float sum_pos = 0.f;
        float sum_neg = 0.f;

        // Vectorized load using float4 for aligned memory access
        int vectorized_end = (feat_size / 4) * 4;
        const float4* anchor_vec = reinterpret_cast<const float4*>(anchor + offset);
        const float4* positive_vec = reinterpret_cast<const float4*>(positive + offset);
        const float4* negative_vec = reinterpret_cast<const float4*>(negative + offset);
        int num_vec = vectorized_end / 4;

        for (int i = 0; i < num_vec; ++i) {
            float4 a = anchor_vec[i];
            float4 p = positive_vec[i];
            float4 n = negative_vec[i];
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
        for (int i = vectorized_end; i < feat_size; ++i) {
            float a = anchor[offset + i];
            float p = positive[offset + i];
            float n = negative[offset + i];
            float d = a - p;
            sum_pos += d * d;
            d = a - n;
            sum_neg += d * d;
        }

        float loss = std::sqrt(sum_pos) - std::sqrt(sum_neg) + margin;
        output[batch_idx] = (loss > 0.f) ? loss : 0.f;
    }
}

torch::Tensor triplet_margin_loss_cpu_optimized(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {

    TORCH_CHECK(!anchor.is_cuda(), "anchor must be a CPU tensor");
    TORCH_CHECK(!positive.is_cuda(), "positive must be a CPU tensor");
    TORCH_CHECK(!negative.is_cuda(), "negative must be a CPU tensor");

    const int batch_size = anchor.size(0);
    const int feat_size = anchor.size(1);
    auto output = torch::empty({batch_size}, anchor.options());

    triplet_margin_loss_cpu(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        output.data_ptr<float>(),
        margin,
        batch_size,
        feat_size);

    return output.mean();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &triplet_margin_loss_cpu_optimized, "Triplet margin loss forward optimized with OpenMP (CPU)");
}
