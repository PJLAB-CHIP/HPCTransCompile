#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// Warp-level reduction function
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ void warp_reduce(float& sum, float& sumsq) {
    sum = warp_reduce_sum(sum);
    sumsq = warp_reduce_sum(sumsq);
}

__global__ void fused_relu_groupnorm_atomic_opt_kernel(
    float* __restrict__ data,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int N, int C, int D, int H, int W,
    int G, float eps)
{
    __shared__ float s_warp_sum[NUM_WARPS];
    __shared__ float s_warp_sumsq[NUM_WARPS];
    __shared__ float s_mean, s_inv_std;

    const int n = blockIdx.x;
    const int g = blockIdx.y;
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    const int channels_per_group = C / G;
    const int c_start = g * channels_per_group;
    const int group_size = channels_per_group * D * H * W;
    const int group_offset = n * C * D * H * W + c_start * D * H * W;

    // Each thread processes multiple elements using vectorized loads
    float4 thread_sum4 = make_float4(0.f, 0.f, 0.f, 0.f);
    float thread_sum = 0.f;
    float thread_sumsq = 0.f;

    // Process elements in chunks of float4
    const int vec_size = 4;
    const int num_vectors = group_size / vec_size;
    const int vectors_per_thread = (num_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float4* data4 = reinterpret_cast<float4*>(data + group_offset);
    
    #pragma unroll 4
    for (int i = 0; i < vectors_per_thread; i++) {
        const int idx = tid + i * BLOCK_SIZE;
        if (idx < num_vectors) {
            float4 val4 = data4[idx];
            
            // Apply ReLU and accumulate
            val4.x = fmaxf(val4.x, 0.f);
            val4.y = fmaxf(val4.y, 0.f);
            val4.z = fmaxf(val4.z, 0.f);
            val4.w = fmaxf(val4.w, 0.f);
            
            data4[idx] = val4;
            
            thread_sum4.x += val4.x;
            thread_sum4.y += val4.y;
            thread_sum4.z += val4.z;
            thread_sum4.w += val4.w;
            
            thread_sumsq += val4.x * val4.x + val4.y * val4.y + 
                          val4.z * val4.z + val4.w * val4.w;
        }
    }

    // Handle remaining elements
    const int remaining_start = num_vectors * vec_size;
    #pragma unroll 4
    for (int i = tid; i < group_size - remaining_start; i += BLOCK_SIZE) {
        const int idx = group_offset + remaining_start + i;
        float val = data[idx];
        val = fmaxf(val, 0.f);
        data[idx] = val;
        thread_sum += val;
        thread_sumsq += val * val;
    }

    // Combine vector and scalar sums
    float warp_sum = thread_sum4.x + thread_sum4.y + thread_sum4.z + thread_sum4.w + thread_sum;
    float warp_sumsq = thread_sumsq;

    // First level: warp-level reduction
    warp_reduce(warp_sum, warp_sumsq);

    // Store warp results
    if (lane == 0) {
        atomicAdd(&s_warp_sum[wid], warp_sum);
        atomicAdd(&s_warp_sumsq[wid], warp_sumsq);
    }
    __syncthreads();

    // Second level: reduce across warps
    if (wid == 0) {
        warp_sum = (lane < NUM_WARPS) ? s_warp_sum[lane] : 0.f;
        warp_sumsq = (lane < NUM_WARPS) ? s_warp_sumsq[lane] : 0.f;
        
        warp_reduce(warp_sum, warp_sumsq);

        if (lane == 0) {
            float mean = warp_sum / group_size;
            float variance = warp_sumsq / group_size - mean * mean;
            s_mean = mean;
            s_inv_std = rsqrtf(variance + eps);
        }
    }
    __syncthreads();

    // Normalize using the computed statistics
    const float mean = s_mean;
    const float inv_std = s_inv_std;

    #pragma unroll 4
    for (int i = 0; i < vectors_per_thread; i++) {
        const int idx = tid + i * BLOCK_SIZE;
        if (idx < num_vectors) {
            float4 val4 = data4[idx];
            const int base_idx = idx * vec_size;
            
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int channel_idx = (base_idx + j) / (D * H * W);
                const int c = c_start + channel_idx;
                float* val_ptr = &((&val4.x)[j]);
                *val_ptr = (*val_ptr - mean) * inv_std;
                *val_ptr = *val_ptr * __ldg(&gamma[c]) + __ldg(&beta[c]);
            }
            
            data4[idx] = val4;
        }
    }

    // Handle remaining elements
    #pragma unroll 4
    for (int i = tid; i < group_size - remaining_start; i += BLOCK_SIZE) {
        const int idx = group_offset + remaining_start + i;
        const int channel_idx = (remaining_start + i) / (D * H * W);
        const int c = c_start + channel_idx;
        
        float val = data[idx];
        val = (val - mean) * inv_std;
        val = val * __ldg(&gamma[c]) + __ldg(&beta[c]);
        data[idx] = val;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_transpose,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias,
    int64_t groups,
    double eps) {

    auto y = at::conv_transpose3d(
        x,
        conv_transpose,
        /*bias=*/c10::nullopt,
        /*stride=*/{1, 1, 1},
        /*padding=*/{0, 0, 0},
        /*output_padding=*/{0, 0, 0},
        /*groups=*/1,
        /*dilation=*/{1, 1, 1}
    );

    int N = y.size(0);
    int C = y.size(1);
    int D = y.size(2);
    int H = y.size(3);
    int W = y.size(4);
    int G = groups;

    dim3 grid(N, G);
    dim3 block(BLOCK_SIZE);

    fused_relu_groupnorm_atomic_opt_kernel<<<grid, block>>>(
         y.data_ptr<float>(),
         group_norm_weight.data_ptr<float>(),
         group_norm_bias.data_ptr<float>(),
         N, C, D, H, W,
         G, static_cast<float>(eps)
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused ConvTranspose3D + ReLU + GroupNorm with optimized atomic operations (CUDA)");
}