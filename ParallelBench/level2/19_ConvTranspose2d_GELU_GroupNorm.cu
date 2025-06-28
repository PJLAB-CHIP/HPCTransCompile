#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <cmath>

// Kernel: Each block processes one group from the fused convTranspose output.
// Workload is distributed evenly by dynamically choosing the number of threads per block
// based on the group size. Grid-stride loops and optional vectorized loads ensure balanced work.

__global__ void fused_gelu_group_norm_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int group_size,       // = channels_per_group * (H*W)
    int hw,               // H * W
    int channels_per_group,
    int C,                // Total channels
    int num_groups,
    float eps,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias) {

    // Each block processes one group. Calculate group indices.
    int group_global = blockIdx.x; // global group index
    int n = group_global / num_groups;  // batch index
    int g = group_global % num_groups;  // group index
    int base = n * C * hw + g * channels_per_group * hw;  // starting offset for this group

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    int tid = threadIdx.x;
    int block_stride = blockDim.x;

    // Check if group_size is vectorizable: process 4 elements at a time if group_size is divisible by 4
    bool use_vector = (group_size % 4 == 0);
    if (use_vector) {
        const float4* in_vec = reinterpret_cast<const float4*>(in + base);
        float4* out_vec = reinterpret_cast<float4*>(out + base);
        int vec_count = group_size / 4;
        for (int idx = tid; idx < vec_count; idx += block_stride) {
            float4 vals = in_vec[idx];
            float4 gelu_vals;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float v = ((float*)&vals)[j];
                float gelu = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
                ((float*)&gelu_vals)[j] = gelu;
                local_sum += gelu;
                local_sum_sq += gelu * gelu;
            }
            out_vec[idx] = gelu_vals;
        }
    } else {
        // Scalar processing if vector load is not applicable
        for (int idx = tid; idx < group_size; idx += block_stride) {
            float v = in[base + idx];
            float gelu = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
            out[base + idx] = gelu;
            local_sum += gelu;
            local_sum_sq += gelu * gelu;
        }
    }

    // Warp-level reduction using shuffle for sum and sum of squares
    int lane = tid & 31;
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }

    // Shared memory to hold per-warp partial sums (reserve space for up to 32 warps)
    __shared__ float smem_sum[32];
    __shared__ float smem_sum_sq[32];
    int warp_id = tid / 32;
    if (lane == 0) {
        smem_sum[warp_id] = local_sum;
        smem_sum_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();

    // Final reduction from warp sums done by thread 0
    float group_mean = 0.0f;
    float group_inv_std = 0.0f;
    if (tid == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float sum_tot = 0.0f;
        float sum_sq_tot = 0.0f;
        for (int i = 0; i < num_warps; i++) {
            sum_tot += smem_sum[i];
            sum_sq_tot += smem_sum_sq[i];
        }
        group_mean = sum_tot / group_size;
        float variance = sum_sq_tot / group_size - group_mean * group_mean;
        group_inv_std = rsqrtf(variance + eps);
        smem_sum[0] = group_mean;   // reuse shared memory to broadcast
        smem_sum[1] = group_inv_std;
    }
    __syncthreads();

    group_mean = smem_sum[0];
    group_inv_std = smem_sum[1];

    // Normalize and apply affine transformation with grid-stride loop
    if (use_vector) {
        float4* out_vec = reinterpret_cast<float4*>(out + base);
        int vec_count = group_size / 4;
        for (int idx = tid; idx < vec_count; idx += block_stride) {
            float4 vals = out_vec[idx];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float gelu = ((float*)&vals)[j];
                float norm = (gelu - group_mean) * group_inv_std;
                // Compute channel index: each channel has 'hw' elements
                int k = idx * 4 + j; // overall element index within the group
                int ch = k / hw;  // channel index within the group
                int global_ch = g * channels_per_group + ch;  // global channel index for group norm params
                float alpha = gn_weight[global_ch];
                float beta = gn_bias[global_ch];
                ((float*)&vals)[j] = norm * alpha + beta;
            }
            out_vec[idx] = vals;
        }
    } else {
        for (int idx = tid; idx < group_size; idx += block_stride) {
            float gelu = out[base + idx];
            float norm = (gelu - group_mean) * group_inv_std;
            int ch = idx / hw;
            int global_ch = g * channels_per_group + ch;
            out[base + idx] = norm * gn_weight[global_ch] + gn_bias[global_ch];
        }
    }
}


torch::Tensor forward(
    torch::Tensor x,
    int64_t stride,
    torch::Tensor conv_transpose_weight,
    torch::Tensor conv_transpose_bias,
    torch::Tensor group_norm_weight,
    torch::Tensor group_norm_bias,
    int64_t num_groups) {

    // Ensure tensors are contiguous and on CUDA
    x = x.contiguous();
    conv_transpose_weight = conv_transpose_weight.contiguous();
    conv_transpose_bias = conv_transpose_bias.contiguous();
    group_norm_weight = group_norm_weight.contiguous();
    group_norm_bias = group_norm_bias.contiguous();

    if (!x.is_cuda()) x = x.cuda();
    if (!conv_transpose_weight.is_cuda()) conv_transpose_weight = conv_transpose_weight.cuda();
    if (!conv_transpose_bias.is_cuda()) conv_transpose_bias = conv_transpose_bias.cuda();
    if (!group_norm_weight.is_cuda()) group_norm_weight = group_norm_weight.cuda();
    if (!group_norm_bias.is_cuda()) group_norm_bias = group_norm_bias.cuda();

    // Perform transposed convolution
    auto conv_out = at::conv_transpose2d(x, conv_transpose_weight, conv_transpose_bias, {stride});
    auto output = at::empty_like(conv_out);

    int N = conv_out.size(0);
    int C = conv_out.size(1);
    int H = conv_out.size(2);
    int W = conv_out.size(3);
    int hw = H * W;
    int channels_per_group = C / num_groups;
    int group_size = channels_per_group * hw;

    // Dynamically determine block size to evenly distribute the workload for each group
    int threads = (group_size < 256) ? ((group_size < 32) ? 32 : group_size) : 256;
    int total_groups = N * num_groups;

    int shared_mem_size = 64 * sizeof(float); // Allocate enough shared memory for warp reductions

    // Launch one block per group
    fused_gelu_group_norm_kernel<<<total_groups, threads, shared_mem_size>>>(
        conv_out.data_ptr<float>(),
        output.data_ptr<float>(),
        group_size,
        hw,
        channels_per_group,
        C,
        num_groups,
        1e-5f,
        group_norm_weight.data_ptr<float>(),
        group_norm_bias.data_ptr<float>()
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused ConvTranspose2d with GELU+GroupNorm with Even Workload Distribution (CUDA)");
}
