#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <limits>

#define BLOCK_SIZE 32

__global__ void shared_bias_kernel(
    float* __restrict__ att,
    const float* __restrict__ bias,
    int64_t B,
    int64_t n_head,
    int64_t T,
    float scale,
    float fill_value
) {
    __shared__ float bias_tile[BLOCK_SIZE][BLOCK_SIZE];

    int tile_i = blockIdx.y * BLOCK_SIZE;
    int tile_j = blockIdx.x * BLOCK_SIZE;

    int local_i = threadIdx.y;
    int local_j = threadIdx.x;

    int global_i = tile_i + local_i;
    int global_j = tile_j + local_j;

    // Load bias tile into shared memory
    if (global_i < T && global_j < T) {
        bias_tile[local_i][local_j] = bias[global_i * T + global_j];
    }
    __syncthreads();
    __threadfence_block();

    // Process all batches and heads for this (i,j) tile
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < n_head; ++h) {
            if (global_i < T && global_j < T) {
                int64_t idx = b * n_head * T * T + h * T * T + global_i * T + global_j;
                float val = att[idx] * scale;
                bool is_masked = (bias_tile[local_i][local_j] == 0.0f);
                val = is_masked ? fill_value : val;
                att[idx] = fmaxf(val, 0.0f);
            }
        }
    }
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor c_attn_weight,
    at::Tensor c_attn_bias,
    at::Tensor bias,
    int64_t n_head,
    int64_t n_embd
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    
    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t C = x.size(2);
    const int64_t hs = C / n_head;
    const float scale = 1.0f / sqrtf(hs);

    // Compute qkv projections
    at::Tensor qkv = at::addmm(c_attn_bias, x.view({B*T, C}), c_attn_weight.t()).view({B, T, 3*C});
    
    // Split and reshape q,k,v
    auto chunks = qkv.split({C, C, C}, 2);
    at::Tensor q = chunks[0].view({B, T, n_head, hs}).permute({0, 2, 1, 3});
    at::Tensor k = chunks[1].view({B, T, n_head, hs}).permute({0, 2, 1, 3});
    
    // Compute attention matrix
    at::Tensor att = at::matmul(q, k.transpose(-2, -1)).contiguous();

    // Prepare bias slice
    at::Tensor bias_slice = bias.slice(2, 0, T).slice(3, 0, T).contiguous();
    const float* bias_data = bias_slice.data_ptr<float>();

    // Kernel launch configuration
    dim3 grid((T + BLOCK_SIZE - 1) / BLOCK_SIZE, (T + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    shared_bias_kernel<<<grid, block>>>(
        att.data_ptr<float>(),
        bias_data,
        B,
        n_head,
        T,
        scale,
        -std::numeric_limits<float>::infinity()
    );
    cudaDeviceSynchronize();

    // Final matmul and reshape
    return at::matmul(att, chunks[2].view({B, T, n_head, hs}).permute({0, 2, 1, 3}))
           .permute({0, 2, 1, 3}).reshape({B, T, C});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared Bias 50_ReLUSelfAttention");
}