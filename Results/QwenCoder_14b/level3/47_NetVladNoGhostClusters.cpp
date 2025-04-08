#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace at;

// Define the fused_assignment_kernel function for CPU
void fused_assignment_kernel_cpu(
    const Tensor& x,
    const Tensor& clusters,
    const Tensor& bn_weight,
    const Tensor& bn_bias,
    const Tensor& bn_mean,
    const Tensor& bn_var,
    Tensor& output,
    int64_t start_idx,
    int64_t chunk_size,
    int64_t D,
    int64_t KplusG,
    bool is_training) {
    
    #pragma omp parallel for collapse(2)
    for (int64_t row = start_idx; row < start_idx + chunk_size; ++row) {
        float sum[KplusG];
        #pragma omp simd
        for (int64_t col = 0; col < KplusG; ++col) {
            sum[col] = 0.0f;
            for (int64_t i = 0; i < D; ++i) {
                sum[col] += x[row * D + i] * clusters[i * KplusG + col];
            }
        }

        // Apply Batch Normalization
        for (int64_t col = 0; col < KplusG; ++col) {
            float val = sum[col];
            if (!is_training) {
                val = (val - bn_mean[col].item<float>()) * bn_weight[col].item<float>() / std::sqrt(bn_var[col].item<float>() + 1e-5f) + bn_bias[col].item<float>();
            }
            output[row * KplusG + col] = val;
        }

        // Softmax reduction
        float max_val = -INFINITY;
        for (int64_t col = 0; col < KplusG; ++col) {
            max_val = std::max(max_val, output[row * KplusG + col]);
        }

        float sum_exp = 0.0f;
        for (int64_t col = 0; col < KplusG; ++col) {
            output[row * KplusG + col] = std::exp(output[row * KplusG + col] - max_val);
            sum_exp += output[row * KplusG + col];
        }

        for (int64_t col = 0; col < KplusG; ++col) {
            output[row * KplusG + col] /= sum_exp;
        }
    }
}

// Define the forward function for CPU
Tensor forward_cpu(
    Tensor x,
    Tensor clusters,
    Tensor clusters2,
    Tensor bn_weight,
    Tensor bn_bias,
    Tensor bn_running_mean,
    Tensor bn_running_var,
    int64_t feature_size,
    int64_t cluster_size,
    bool is_training) {
    
    int64_t B = x.size(0);
    int64_t N = x.size(1);
    int64_t D = feature_size;
    int64_t K = cluster_size;
    int64_t KplusG = clusters.size(1);
    int64_t BxN = B * N;

    x = x.reshape({-1, D});
    auto assignment = torch::empty({BxN, KplusG}, x.options());

    // Process data in chunks
    for (int64_t chunk_start = 0; chunk_start < BxN; chunk_start += 1024) {
        int64_t current_chunk_size = std::min(static_cast<int64_t>(1024), BxN - chunk_start);
        fused_assignment_kernel_cpu(
            x,
            clusters,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            assignment,
            chunk_start,
            current_chunk_size,
            D,
            KplusG,
            is_training);
    }

    assignment = assignment.narrow(1, 0, K).reshape({B, N, K});
    auto a_sum = assignment.sum(1, true);
    clusters2 = clusters2.expand({B, D, K});
    auto a = clusters2 * a_sum;

    assignment = assignment.transpose(1, 2);
    x = x.reshape({B, N, D});
    auto vlad = torch::bmm(assignment, x).transpose(1, 2) - a;

    vlad = torch::nn::functional::normalize(
        vlad, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    vlad = vlad.reshape({B, D * K});
    vlad = torch::nn::functional::normalize(
        vlad, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    return vlad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cpu, "NetVLAD forward on CPU");
}