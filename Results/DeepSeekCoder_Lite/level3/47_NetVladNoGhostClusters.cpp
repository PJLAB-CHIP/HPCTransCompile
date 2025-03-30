#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <omp.h>
#include <cmath>
#include <cfloat>

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cpu(), #x " must be a CPU tensor")

constexpr int TILE_SIZE = 128;
constexpr int NUM_STREAMS = 2;
constexpr int CHUNK_SIZE = 1024;

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor clusters,
    torch::Tensor clusters2,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    int64_t feature_size,
    int64_t cluster_size,
    bool is_training) {
    
    CHECK_INPUT(x);
    CHECK_INPUT(clusters);
    CHECK_INPUT(clusters2);
    CHECK_INPUT(bn_weight);
    CHECK_INPUT(bn_bias);
    CHECK_INPUT(bn_running_mean);
    CHECK_INPUT(bn_running_var);

    int64_t B = x.size(0);
    int64_t N = x.size(1);
    int64_t D = feature_size;
    int64_t K = cluster_size;
    int64_t KplusG = clusters.size(1);
    int64_t BxN = B * N;

    x = x.reshape({-1, D});
    auto assignment = torch::empty({BxN, KplusG}, x.options());

    int64_t chunk_start = 0;
    int64_t current_chunk_size = std::min(static_cast<int64_t>(CHUNK_SIZE), BxN - chunk_start);

    // Process data in chunks using multiple threads
    #pragma omp parallel for
    for (int64_t chunk_start = 0; chunk_start < BxN; chunk_start += CHUNK_SIZE) {
        int64_t current_chunk_size = std::min(static_cast<int64_t>(CHUNK_SIZE), BxN - chunk_start);
        int64_t row = omp_get_thread_num();
        int64_t start_idx = chunk_start;
        int64_t chunk_size = current_chunk_size;

        std::vector<float> smem(TILE_SIZE * TILE_SIZE * 3);
        std::vector<float> smem_max(TILE_SIZE);
        std::vector<float> smem_sum(TILE_SIZE);

        for (int64_t row = start_idx; row < start_idx + chunk_size; row++) {
            float sum = 0.0f;
            for (int64_t i = 0; i < D; i++) {
                sum += x[row * D + i] * clusters[i * KplusG + col];
            }
            smem[col] = sum;

            if (!is_training) {
                smem[col] = (smem[col] - bn_running_mean[col]) * bn_weight[col] / sqrtf(bn_running_var[col] + 1e-5f) + bn_bias[col];
            }

            float max_val = -INFINITY;
            for (int64_t i = 0; i < KplusG; i++) {
                max_val = fmaxf(max_val, smem[i]);
            }
            smem_max[tid] = max_val;

            for (int64_t s = TILE_SIZE / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    smem_max[tid] = fmaxf(smem_max[tid], smem_max[tid + s]);
                }
            }

            float sum_exp = 0.0f;
            smem[col] = __expf(smem[col] - max_val);
            for (int64_t i = 0; i < KplusG; i++) {
                sum_exp += __expf(smem[i] - max_val);
            }
            smem_sum[tid] = sum_exp;

            for (int64_t s = TILE_SIZE / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    smem_sum[tid] += smem_sum[tid + s];
                }
            }

            output[row * KplusG + col] = smem[col] / smem_sum[0];
        }
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
    m.def("forward", &forward, "NetVLAD forward with streams");
}