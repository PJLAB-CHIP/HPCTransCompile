#include <torch/extension.h>
#include <cmath>
#include <omp.h>

// Function to compute cosine similarity loss
float compute_cosine_similarity_loss(const float* predictions, const float* targets, int N, int D) {
    float total_loss = 0.0f;
    #pragma omp parallel for reduction(+:total_loss)
    for (int row = 0; row < N; ++row) {
        float sum_dot = 0.0f;
        float sum_pred_sq = 0.0f;
        float sum_target_sq = 0.0f;

        for (int i = 0; i < D; ++i) {
            float p = predictions[row * D + i];
            float t = targets[row * D + i];
            sum_dot += p * t;
            sum_pred_sq += p * p;
            sum_target_sq += t * t;
        }

        const float eps = 1e-8f;
        float norm_pred = std::sqrt(sum_pred_sq);
        float norm_target = std::sqrt(sum_target_sq);
        float denominator = norm_pred * norm_target;
        denominator = std::fmaxf(denominator, eps);
        float cos_sim = sum_dot / denominator;
        total_loss += (1.0f - cos_sim);
    }

    return total_loss / N;
}

// Host binding function
torch::Tensor blocksize_tuning_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);
    float loss = compute_cosine_similarity_loss(predictions.data_ptr<float>(), targets.data_ptr<float>(), N, D);
    auto output = torch::tensor({loss}, predictions.options());

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &blocksize_tuning_cosine_similarity_loss_forward, "Blocksize Tuning Cosine Similarity Loss Forward (CPU)");
}