#include <torch/extension.h>
#include <math.h>
#include <omp.h>

// Host binding function with block size dispatching
torch::Tensor blocksize_tuning_cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);
    auto output = torch::zeros({1}, predictions.options());

    // Experiment with a range of block sizes based on the D dimension
    if (D <= 64) {
        #pragma omp parallel for
        for (int row = 0; row < N; ++row) {
            float sum_dot = 0.0f;
            float sum_pred_sq = 0.0f;
            float sum_target_sq = 0.0f;

            // Iterate over the D dimension in strides of BLOCK_SIZE
            for (int i = 0; i < D; i += 32) {
                for (int tid = 0; tid < 32; ++tid) {
                    int idx = row * D + i + tid;
                    float p = predictions[idx];
                    float t = targets[idx];
                    sum_dot += p * t;
                    sum_pred_sq += p * p;
                    sum_target_sq += t * t;
                }
            }

            // Warp-level reduction using shuffle within each warp
            float final_dot = 0.0f;
            float final_pred_sq = 0.0f;
            float final_target_sq = 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) {
                final_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
                final_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
                final_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
            }

            const float eps = 1e-8f;
            float norm_pred = sqrtf(final_pred_sq);
            float norm_target = sqrtf(final_target_sq);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);
            float cos_sim = final_dot / denominator;
            // Accumulate loss over rows and average by dividing by N
            output[0] += (1.0f - cos_sim) / N;
        }
    } else if (D <= 128) {
        #pragma omp parallel for
        for (int row = 0; row < N; ++row) {
            float sum_dot = 0.0f;
            float sum_pred_sq = 0.0f;
            float sum_target_sq = 0.0f;

            // Iterate over the D dimension in strides of BLOCK_SIZE
            for (int i = 0; i < D; i += 64) {
                for (int tid = 0; tid < 64; ++tid) {
                    int idx = row * D + i + tid;
                    float p = predictions[idx];
                    float t = targets[idx];
                    sum_dot += p * t;
                    sum_pred_sq += p * p;
                    sum_target_sq += t * t;
                }
            }

            // Warp-level reduction using shuffle within each warp
            float final_dot = 0.0f;
            float final_pred_sq = 0.0f;
            float final_target_sq = 0.0f;
            for (int offset = 32; offset > 0; offset /= 2) {
                final_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
                final_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
                final_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
            }

            const float eps = 1e-8f;
            float norm_pred = sqrtf(final_pred_sq);
            float norm_target = sqrtf(final_target_sq);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);
            float cos_sim = final_dot / denominator;
            // Accumulate loss over rows and average by dividing by N
            output[0] += (1.0f - cos_sim) / N;
        }
    } else if (D <= 256) {
        #pragma omp parallel for
        for (int row = 0; row < N; ++row) {
            float sum_dot = 0.0f;
            float sum_pred_sq = 0.0f;
            float sum_target_sq = 0.0f;

            // Iterate over the D dimension in strides of BLOCK_SIZE
            for (int i = 0; i < D; i += 128) {
                for (int tid = 0; tid < 128; ++tid) {
                    int idx = row * D + i + tid;
                    float p = predictions[idx];
                    float t = targets[idx];
                    sum_dot += p * t;
                    sum_pred_sq += p * p;
                    sum_target_sq += t * t;
                }
            }

            // Warp-level reduction using shuffle within each warp
            float final_dot = 0.0f;
            float final_pred_sq = 0.0f;
            float final_target_sq = 0.0f;
            for (int offset = 64; offset > 0; offset /= 2) {
                final_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
                final_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
                final_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
            }

            const float eps = 1e-8f;
            float norm_pred = sqrtf(final_pred_sq);
            float norm_target = sqrtf(final_target_sq);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);
            float cos_sim = final_dot / denominator;
            // Accumulate loss over rows and average by dividing by N
            output[0] += (1.0f - cos_sim) / N;
        }
    } else if (D <= 512) {
        #pragma omp parallel for
        for (int row = 0; row < N; ++row) {
            float sum_dot = 0.0f;
            float sum_pred_sq = 0.0f;
            float sum_target_sq = 0.0f;

            // Iterate over the D dimension in strides of BLOCK_SIZE
            for (int i = 0; i < D; i += 256) {
                for (int tid = 0; tid < 256; ++tid) {
                    int idx = row * D + i + tid;
                    float p = predictions[idx];
                    float t = targets[idx];
                    sum_dot += p * t;
                    sum_pred_sq += p * p;
                    sum_target_sq += t * t;
                }
            }

            // Warp-level reduction using shuffle within each warp
            float final_dot = 0.0f;
            float final_pred_sq = 0.0f;
            float final_target_sq = 0.0f;
            for (int offset = 128; offset > 0; offset /= 2) {
                final_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
                final_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
                final_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
            }

            const float eps = 1e-8f;
            float norm_pred = sqrtf(final_pred_sq);
            float norm_target = sqrtf(final_target_sq);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);
            float cos_sim = final_dot / denominator;
            // Accumulate loss over rows and average by dividing by N
            output[0] += (1.0f - cos_sim) / N;
        }
    } else {
        #pragma omp parallel for
        for (int row = 0; row < N; ++row) {
            float sum_dot = 0.0f;
            float sum_pred_sq = 0.0f;
            float sum_target_sq = 0.0f;

            // Iterate over the D dimension in strides of BLOCK_SIZE
            for (int i = 0; i < D; i += 512) {
                for (int tid = 0; tid < 512; ++tid) {
                    int idx = row * D + i + tid;
                    float p = predictions[idx];
                    float t = targets[idx];
                    sum_dot += p * t;
                    sum_pred_sq += p * p;
                    sum_target_sq += t * t;
                }
            }

            // Warp-level reduction using shuffle within each warp
            float final_dot = 0.0f;
            float final_pred_sq = 0.0f;
            float final_target_sq = 0.0f;
            for (int offset = 256; offset > 0; offset /= 2) {
                final_dot += __shfl_down_sync(0xffffffff, sum_dot, offset);
                final_pred_sq += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
                final_target_sq += __shfl_down_sync(0xffffffff, sum_target_sq, offset);
            }

            const float eps = 1e-8f;
            float norm_pred = sqrtf(final_pred_sq);
            float norm_target = sqrtf(final_target_sq);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);
            float cos_sim = final_dot / denominator;
            // Accumulate loss over rows and average by dividing by N
            output[0] += (1.0f - cos_sim) / N;
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &blocksize_tuning_cosine_similarity_loss_forward, "Blocksize Tuning Cosine Similarity Loss Forward (CPU)");
}