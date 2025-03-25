#include <torch/extension.h>
#include <ATen/ATen.h>
#include <omp.h>
#include <cmath>
#include <limits>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

constexpr int TILE_SIZE = 128;
constexpr int NUM_THREADS = 4;

void fused_assignment_cpu(
    const float* x,
    const float* clusters,
    const float* bn_weight,
    const float* bn_bias,
    const float* bn_mean,
    const float* bn_var,
    float* output,
    int64_t start_idx,
    int64_t chunk_size,
    int64_t D,
    int64_t KplusG,
    bool is_training) {
    
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int64_t row = start_idx; row < start_idx + chunk_size; ++row) {
        float smem[KplusG];
        float smem_max[KplusG];
        float smem_sum[KplusG];
        
        // Initialize shared memory arrays
        for (int i = 0; i < KplusG; ++i) {
            smem[i] = 0.0f;
            smem_max[i] = -std::numeric_limits<float>::infinity();
            smem_sum[i] = 0.0f;
        }
        
        // Compute matmul row
        for (int i = 0; i < D; ++i) {
            smem[i] += x[row * D + i] * clusters[i * KplusG + i];
        }
        
        // Apply BN
        for (int i = 0; i < KplusG; ++i) {
            float val = smem[i];
            if (!is_training) {
                val = (val - bn_mean[i]) * bn_weight[i] / std::sqrt(bn_var[i] + 1e-5f) + bn_bias[i];
            }
            smem[i] = val;
        }
        
        // Softmax reduction with improved memory access pattern
        for (int i = 0; i < KplusG; ++i) {
            smem_max[i] = fmaxf(smem_max[i], smem[i]);
        }
        
        // Reduce max values
        for (int s = KplusG / 2; s > 0; s >>= 1) {
            for (int i = 0; i < KplusG; ++i) {
                if (i + s < KplusG) {
                    smem_max[i] = fmaxf(smem_max[i], smem_max[i + s]);
                }
            }
        }
        
        float max_val = smem_max[0];
        
        // Compute softmax
        for (int i = 0; i < KplusG; ++i) {
            smem_sum[i] = expf(smem[i] - max_val);
        }
        
        // Reduce sum values
        for (int s = KplusG / 2; s > 0; s >>= 1) {
            for (int i = 0; i < KplusG; ++i) {
                if (i + s < KplusG) {
                    smem_sum[i] += smem_sum[i + s];
                }
            }
        }
        
        output[row * KplusG] = smem[0] / smem_sum[0];
    }
}

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

    // Process data in chunks using multiple threads
    for (int64_t chunk_start = 0; chunk_start < BxN; chunk_start += TILE_SIZE) {
        int64_t current_chunk_size = std::min(static_cast<int64_t>(TILE_SIZE), BxN - chunk_start);
        
        fused_assignment_cpu(
            x.data_ptr<float>(),
            clusters.data_ptr<float>(),
            bn_weight.data_ptr<float>(),
            bn_bias.data_ptr<float>(),
            bn_running_mean.data_ptr<float>(),
            bn_running_var.data_ptr<float>(),
            assignment.data_ptr<float>(),
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
    m.def("forward", &forward, "NetVLAD forward with OpenMP");
}