#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <cmath>
#include <omp.h>

// Vectorized elementwise addition kernel (same as previous optimized version)
void elementwise_add_vectorized_kernel(
    float* __restrict__ out,
    const float* __restrict__ sum_weight,
    int64_t n_vec
) {
    const float add_val = *sum_weight;
    float4* out_vec = reinterpret_cast<float4*>(out);

    #pragma omp parallel for
    for(int idx = 0; idx < n_vec; ++idx) {
        float4 vals = out_vec[idx];
        vals.x += add_val;
        vals.y += add_val;
        vals.z += add_val;
        vals.w += add_val;
        out_vec[idx] = vals;
    }
}

// Vectorized elementwise addition kernel (same as previous optimized version)
void elementwise_add_tail_kernel(
    float* __restrict__ out,
    const float* __restrict__ sum_weight,
    int64_t n_rem
) {
    const float add_val = *sum_weight;
    #pragma omp parallel for
    for(int idx = 0; idx < n_rem; ++idx) {
        out[idx] += add_val;
    }
}

// Custom layer norm kernel with shared memory reductions
void layer_norm_forward_kernel(
    const float* __restrict__ X,
    float* __restrict__ Y,
    const float* gamma,
    const float* beta,
    int num_features,
    int feature_stride,
    float epsilon
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Compute feature offset
    const int feature_offset = bid * feature_stride;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    // Each thread computes partial sums
    for (int i = tid; i < num_features; i += blockDim.x) {
        float val = X[feature_offset + i];
        sum += val;
        sum_sq += val * val;
    }

    // Block-wide reduction for sum and sum_sq
    __shared__ float block_sum[32];
    __shared__ float block_sum_sq[32];
    
    // Reduce within warp using shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    
    // First thread in warp stores to shared memory
    if (tid % 32 == 0) {
        block_sum[tid / 32] = sum;
        block_sum_sq[tid / 32] = sum_sq;
    }
    __syncthreads();
    
    // Final reduction across warps in block
    if (tid < 32) {
        float warp_sum = (tid < blockDim.x / 32) ? block_sum[tid] : 0.0f;
        float warp_sum_sq = (tid < blockDim.x / 32) ? block_sum_sq[tid] : 0.0f;
        
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
            warp_sum_sq += __shfl_down_sync(0xffffffff, warp_sum_sq, offset);
        }
        
        if (tid == 0) {
            block_sum[0] = warp_sum;
            block_sum_sq[0] = warp_sum_sq;
        }
    }
    __syncthreads();
    
    const float mean = block_sum[0] / num_features;
    const float var = block_sum_sq[0] / num_features - mean * mean;
    const float inv_std = 1.0f / sqrt(var + epsilon);
    
    // Apply normalization
    for (int i = tid; i < num_features; i += blockDim.x) {
        const int idx = feature_offset + i;
        float val = (X[idx] - mean) * inv_std;
        Y[idx] = val * gamma[i] + beta[i];
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor conv_transpose_weight,
    torch::Tensor conv_transpose_bias,
    torch::Tensor sum_weight,
    torch::Tensor norm_weight,
    torch::Tensor norm_bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    std::vector<int64_t> pool_kernel_size,
    std::vector<int64_t> norm_shape
) {
    at::IntArrayRef strideRef(stride);
    at::IntArrayRef paddingRef(padding);
    at::IntArrayRef outputPaddingRef(output_padding);
    at::IntArrayRef poolKernelRef(pool_kernel_size);
    at::IntArrayRef normShapeRef(norm_shape);

    // 1. 3D transposed convolution
    auto out = at::conv_transpose3d(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        strideRef,
        paddingRef,
        outputPaddingRef,
        /*groups=*/1,
        /*dilation=*/1
    );

    // 2. Optimized elementwise addition
    int64_t num_elements = out.numel();
    const int block_size = 256;
    int64_t n_vec = num_elements / 4;
    int64_t rem = num_elements % 4;

    if (n_vec > 0) {
        int grid_size = (n_vec + block_size - 1) / block_size;
        elementwise_add_vectorized_kernel(
            out.data_ptr<float>(),
            sum_weight.data_ptr<float>(),
            n_vec
        );
    }
    if (rem > 0) {
        int grid_size = (rem + block_size - 1) / block_size;
        float* tail_ptr = out.data_ptr<float>() + n_vec * 4;
        elementwise_add_tail_kernel(
            tail_ptr,
            sum_weight.data_ptr<float>(),
            rem
        );
    }

    // 3. Custom layer normalization
    const int num_features = norm_shape.back();
    const int feature_stride = num_features;
    const int num_blocks = out.numel() / num_features;
    
    const int threads_per_block = 128;
    size_t shared_mem_size = 2 * (threads_per_block / 32 + 1) * sizeof(float);
    
    layer_norm_forward_kernel(
        out.data_ptr<float>(),
        out.data_ptr<float>(),
        norm_weight.data_ptr<float>(),
        norm_bias.data_ptr<float>(),
        num_features,
        feature_stride,
        1e-5
    );

    // 4. 3D average pooling
    out = at::avg_pool3d(
        out,
        poolKernelRef,
        poolKernelRef,
        /*padding=*/{0, 0, 0},
        /*ceil_mode=*/false,
        /*count_include_pad=*/true
    );

    // 5. GELU activation
    out = at::gelu(out);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "module_fn forward (CPU)");
}