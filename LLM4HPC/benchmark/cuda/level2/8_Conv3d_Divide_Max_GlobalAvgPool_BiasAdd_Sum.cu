#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cfloat>

// Tunable parameters for the fused kernel and reduction
#define BLOCK_SIZE_FUSED 256        // for fused division, max pooling, and avg pooling kernel
#define BLOCK_SIZE_REDUCTION 256    // for the reduction kernel (optimized with warp-level unrolling)

// Fused kernel: performs division, 3D max pooling over non-overlapping windows,
// and then adaptive average pooling (summing over all pooled windows) with bias addition.
// Input:
//   in         : Pointer to conv3d output (shape: N x C x D x H x W)
//   out        : Pointer to output tensor (shape: N x C) containing the average pooled results + bias
//   N, C, D, H, W: dimensions of conv3d output
//   poolD, poolH, poolW: dimensions of the pooling window
//   OD, OH, OW : number of pooling windows in each spatial dimension
//   divisor    : Division factor to be applied (using multiplication by reciprocal)
//   bias       : Bias pointer (assumed shape: C) to be added per channel
__global__ void fused_divide_maxpool_avg_kernel(const float* __restrict__ in,
                                                  float* __restrict__ out,
                                                  int N, int C,
                                                  int D, int H, int W,
                                                  int poolD, int poolH, int poolW,
                                                  int OD, int OH, int OW,
                                                  float divisor,
                                                  const float* __restrict__ bias) {
    // Each block is responsible for one (n, c) pair
    int n = blockIdx.x;
    int c = blockIdx.y;

    // Total number of pooling windows for this (n, c)
    int total_windows = OD * OH * OW;

    float partialSum = 0.0f;
    // Each thread processes a subset of pooling windows in a grid-stride loop
    for (int idx = threadIdx.x; idx < total_windows; idx += blockDim.x) {
        // Decode linear index into pooling window coordinates (od, oh, ow)
        int ow = idx % OW;
        int tmp = idx / OW;
        int oh = tmp % OH;
        int od = tmp / OH;  // since tmp = od * OH + oh

        // Determine starting indices in D, H, W for the pooling window
        int d_start = od * poolD;
        int h_start = oh * poolH;
        int w_start = ow * poolW;

        float max_val = -FLT_MAX;
        // Iterate over the pooling window
        #pragma unroll 4
        for (int d = d_start; d < d_start + poolD; ++d) {
            for (int h = h_start; h < h_start + poolH; ++h) {
                for (int w = w_start; w < w_start + poolW; ++w) {
                    // Compute linear index in conv output tensor: shape (N, C, D, H, W)
                    int index = (((n * C + c) * D + d) * H + h) * W + w;
                    float val = in[index] * (1.0f / divisor);
                    max_val = max(max_val, val);
                }
            }
        }
        partialSum += max_val;
    }

    // Use shared memory to reduce partial sums from threads within the block
    __shared__ float sdata[BLOCK_SIZE_FUSED];
    int tid = threadIdx.x;
    sdata[tid] = partialSum;
    __syncthreads();

    // Standard reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Compute adaptive average pooling (divide by total number of pooling windows)
        float avg = sdata[0] / static_cast<float>(total_windows);
        // Add bias for channel c
        out[n * C + c] = avg + bias[c];
    }
}

// Optimized reduction kernel (from snippet 2) to sum the (N, C) tensor along a chosen dimension
// For sum_dim == 1, reduction is over channels (output shape: N)
// For sum_dim == 0, reduction is over batch (output shape: C)
__global__ void reduction_sum_kernel(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      int N, int C, int sum_dim) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    if (sum_dim == 1) {
        // Each block processes one sample (n)
        int n = blockIdx.x;
        float sum = 0.0f;
        for (int c = tid; c < C; c += blockDim.x) {
            sum += in[n * C + c];
        }
        sdata[tid] = sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 32; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid < 32) {
            volatile float* smem = sdata;
            smem[tid] += smem[tid + 32];
            smem[tid] += smem[tid + 16];
            smem[tid] += smem[tid + 8];
            smem[tid] += smem[tid + 4];
            smem[tid] += smem[tid + 2];
            smem[tid] += smem[tid + 1];
        }
        if (tid == 0) {
            out[n] = sdata[0];
        }
    } else if (sum_dim == 0) {
        // Each block processes one channel (c)
        int c = blockIdx.x;
        float sum = 0.0f;
        for (int n = tid; n < N; n += blockDim.x) {
            sum += in[n * C + c];
        }
        sdata[tid] = sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 32; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid+s];
            __syncthreads();
        }
        if (tid < 32) {
            volatile float* smem = sdata;
            smem[tid] += smem[tid + 32];
            smem[tid] += smem[tid + 16];
            smem[tid] += smem[tid + 8];
            smem[tid] += smem[tid + 4];
            smem[tid] += smem[tid + 2];
            smem[tid] += smem[tid + 1];
        }
        if (tid == 0) {
            out[c] = sdata[0];
        }
    }
}

// The forward_cuda function performs:
// 1) 3D convolution (using at::conv3d for correctness),
// 2) a fused kernel that computes division, 3D max pooling across windows, adaptive average pooling, and bias addition,
// 3) a reduction kernel to sum the (N, C) tensor along the specified dimension (sum_dim == 0 or 1).

torch::Tensor forward_cuda(torch::Tensor x,
                             double divisor,
                             std::vector<int64_t> pool_size,
                             int64_t sum_dim,
                             torch::Tensor conv_weight,
                             torch::Tensor conv_bias,
                             torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor.");
    TORCH_CHECK(conv_weight.is_cuda(), "conv_weight must be a CUDA tensor.");
    TORCH_CHECK(conv_bias.is_cuda(), "conv_bias must be a CUDA tensor.");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor.");

    // 1) 3D convolution using PyTorch's conv3d
    auto conv_out = at::conv3d(x, conv_weight, conv_bias);
    // conv_out shape: (N, C, D, H, W)
    int N = conv_out.size(0);
    int C = conv_out.size(1);
    int D = conv_out.size(2);
    int H = conv_out.size(3);
    int W = conv_out.size(4);

    // Pooling window sizes
    int poolD = pool_size[0];
    int poolH = pool_size[1];
    int poolW = pool_size[2];

    // Compute output dimensions for the pooling stage (assumes perfect divisibility)
    int OD = D / poolD;
    int OH = H / poolH;
    int OW = W / poolW;

    auto options = conv_out.options();
    // Output of fused kernel: adaptive average pooling result per (n, c)
    auto avg_out = at::empty({N, C}, options);

    // Launch fused kernel with a 2D grid: one block for each (n, c) pair
    dim3 grid(N, C);
    fused_divide_maxpool_avg_kernel<<<grid, BLOCK_SIZE_FUSED>>>(
        conv_out.data_ptr<float>(),
        avg_out.data_ptr<float>(),
        N, C, D, H, W,
        poolD, poolH, poolW,
        OD, OH, OW,
        static_cast<float>(divisor),
        bias.data_ptr<float>()
    );

    // 3) Reduction: sum over the (N, C) result along an input-specified dimension.
    torch::Tensor final_out;
    if (sum_dim == 1) {
        // Sum over channels; final output shape: (N)
        final_out = at::empty({N}, options);
        reduction_sum_kernel<<<N, BLOCK_SIZE_REDUCTION, BLOCK_SIZE_REDUCTION * sizeof(float)>>>(
            avg_out.data_ptr<float>(),
            final_out.data_ptr<float>(),
            N, C, sum_dim
        );
    } else if (sum_dim == 0) {
        // Sum over batch; final output shape: (C)
        final_out = at::empty({C}, options);
        reduction_sum_kernel<<<C, BLOCK_SIZE_REDUCTION, BLOCK_SIZE_REDUCTION * sizeof(float)>>>(
            avg_out.data_ptr<float>(),
            final_out.data_ptr<float>(),
            N, C, sum_dim
        );
    } else {
        TORCH_CHECK(false, "sum_dim must be 0 or 1");
    }

    return final_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Fused conv3d, divide, max pool, adaptive avg pool, bias add, and reduction kernel");
}
