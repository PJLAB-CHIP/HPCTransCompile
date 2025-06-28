#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

//---------------------------------------------------------------------------
// Fused Matrix Multiplication with Bias Addition Kernel
// Computes: C = A * (B^T) + bias, where A is [M x K], B is [N x K] (stored row-wise),
// and bias is a vector of length N. Uses shared memory tiling for improved performance.
//---------------------------------------------------------------------------
__global__ void FusedMatMulBiasKernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      const float* __restrict__ bias,
                                      float* __restrict__ C,
                                      int M, int N, int K) {
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int tiled_k = t * TILE_SIZE;
        // Load A tile
        if (row < M && (tiled_k + threadIdx.x) < K)
            Asub[threadIdx.y][threadIdx.x] = A[row * K + tiled_k + threadIdx.x];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile (B is stored such that we use its transpose logic)
        if (col < N && (tiled_k + threadIdx.y) < K)
            Bsub[threadIdx.y][threadIdx.x] = B[col * K + tiled_k + threadIdx.y];
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles together
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Write result with bias addition
    if (row < M && col < N) {
        C[row * N + col] = sum + bias[col];
    }
}

//---------------------------------------------------------------------------
// Fused Pooling, Activation, Scaling and Max Reduction Kernel
// Input: the linear output from the previous stage of shape [M x N].
// Operation per row:
//   1. Average Pooling: groups contiguous elements with pool_kernel_size. 
//      (If the group is incomplete at the end, it computes the average over available elements.)
//   2. GELU Activation (using the approximate formula: 0.5 * x * (1 + erf(x * 0.70710678))).
//   3. Scaling by scale_factor.
//   4. Maximum reduction over the pooled/activated values.
// Each block processes one row; multiple threads in a block cooperatively reduce the maximum.
//---------------------------------------------------------------------------
__global__ void FusedPoolActMaxKernel(const float* __restrict__ linear_output,
                                      float* __restrict__ output,
                                      int M, int N,
                                      int pool_kernel_size,
                                      int output_length,
                                      float scale_factor) {
    // One block per row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float local_max = -FLT_MAX;

    // Each thread processes multiple pooling bins using striding
    for (int bin = tid; bin < output_length; bin += blockDim.x) {
        int start = bin * pool_kernel_size;
        float sum = 0.0f;
        int count = 0;
        for (int j = 0; j < pool_kernel_size; j++) {
            int col = start + j;
            if (col < N) {
                sum += linear_output[row * N + col];
                count++;
            }
        }
        float avg = sum / count;  // Average pooling result
        // Apply GELU activation: 0.5 * avg * (1 + erf(avg * 0.70710678))
        float gelu = 0.5f * avg * (1.0f + erff(avg * 0.70710678f));
        // Scale the activated output
        gelu *= scale_factor;
        local_max = fmaxf(local_max, gelu);
    }

    // Reduction within block using shared memory
    extern __shared__ float sdata[];  // Dynamically allocated shared memory
    sdata[tid] = local_max;
    __syncthreads();

    // Parallel reduction (assumes blockDim.x is a power of 2)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // The first thread writes the maximum value for this row
    if (tid == 0) {
        output[row] = sdata[0];
    }
}

//---------------------------------------------------------------------------
// Forward function that chains the fused operations
// Steps:
// 1. Compute linear transformation: linear = x * (weight^T) + bias using a tiled matmul kernel.
// 2. Apply fused average pooling, GELU activation, scaling, and maximum reduction across pooled bins.
//---------------------------------------------------------------------------

torch::Tensor forward(
    torch::Tensor x,
    int pool_kernel_size,
    float scale_factor,
    torch::Tensor weight,
    torch::Tensor bias) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    // Ensure tensors are contiguous
    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    // Dimensions
    int M = x.size(0);        // Batch size (number of rows)
    int K = x.size(1);        // Input features
    int N = weight.size(0);   // Output features (number of rows in weight, since weight is transposed)

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    // Allocate tensor for the linear transformation results
    auto linear_output = torch::empty({M, N}, options);

    // Launch fused matrix multiplication + bias addition kernel
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    FusedMatMulBiasKernel<<<gridDim, blockDim>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        linear_output.data_ptr<float>(),
        M, N, K);

    // Determine pooling output length
    int output_length = (N + pool_kernel_size - 1) / pool_kernel_size;

    // Allocate tensor for final output (one value per batch row)
    auto output = torch::empty({M}, options);

    // Launch fused pooling, activation, scaling, and max reduction kernel
    // One block per row, use 256 threads (or adjust based on output_length)
    int threads = 256;
    size_t sharedMemSize = threads * sizeof(float);
    FusedPoolActMaxKernel<<<M, threads, sharedMemSize>>>(
         linear_output.data_ptr<float>(),
         output.data_ptr<float>(),
         M, N, pool_kernel_size, output_length, scale_factor);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused CUDA forward (MatMul+Bias, Pool, GELU, Scale, Max Reduction)");
}
