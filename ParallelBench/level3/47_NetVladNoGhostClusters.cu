#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

constexpr int TILE_SIZE = 128;
constexpr int NUM_STREAMS = 2;
constexpr int CHUNK_SIZE = 1024;

__global__ void fused_assignment_kernel(
    const float* __restrict__ x,
    const float* __restrict__ clusters,
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
    
    int row = blockIdx.x * blockDim.y + threadIdx.y + start_idx;
    int tid = threadIdx.x;
    int col = threadIdx.y;
    
    if (row >= start_idx + chunk_size) return;
    
    __shared__ float smem[TILE_SIZE];
    __shared__ float smem_max[TILE_SIZE];
    __shared__ float smem_sum[TILE_SIZE];
    
    // Compute matmul row
    float sum = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < D; i += TILE_SIZE) {
        sum += x[row * D + i] * clusters[i * KplusG + col];
    }
    atomicAdd(&smem[col], sum);
    
    __syncthreads();
    
    // Apply BN
    float val = smem[col];
    if (!is_training) {
        val = (val - bn_mean[col]) * bn_weight[col] / sqrtf(bn_var[col] + 1e-5f) + bn_bias[col];
    }
    
    // Softmax reduction with improved memory access pattern
    float max_val = -INFINITY;
    #pragma unroll 4
    for (int i = tid; i < KplusG; i += TILE_SIZE) {
        max_val = fmaxf(max_val, smem[i]);
    }
    smem_max[tid] = max_val;
    
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            smem_max[tid] = fmaxf(smem_max[tid], smem_max[tid + s]);
        }
        __syncthreads();
    }
    
    max_val = smem_max[0];
    
    float sum_exp = 0.0f;
    val = __expf(val - max_val);
    #pragma unroll 4
    for (int i = tid; i < KplusG; i += TILE_SIZE) {
        sum_exp += __expf(smem[i] - max_val);
    }
    smem_sum[tid] = sum_exp;
    
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            smem_sum[tid] += smem_sum[tid + s];
        }
        __syncthreads();
    }
    
    output[row * KplusG + col] = val / smem_sum[0];
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

    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    x = x.reshape({-1, D});
    auto assignment = torch::empty({BxN, KplusG}, x.options());

    dim3 block(TILE_SIZE, TILE_SIZE);
    size_t shared_mem = TILE_SIZE * sizeof(float) * 3; // For smem, smem_max, and smem_sum

    // Process data in chunks using multiple streams
    for (int64_t chunk_start = 0; chunk_start < BxN; chunk_start += CHUNK_SIZE) {
        int64_t current_chunk_size = std::min(static_cast<int64_t>(CHUNK_SIZE), BxN - chunk_start);
        int stream_idx = (chunk_start / CHUNK_SIZE) % NUM_STREAMS;
        
        dim3 grid((current_chunk_size + TILE_SIZE - 1) / TILE_SIZE);
        
        fused_assignment_kernel<<<grid, block, shared_mem, streams[stream_idx]>>>(
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

    // Synchronize all streams before proceeding
    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
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