#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <cfloat>
#include <omp.h>

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

//---------------------------------------------------------------------------
// Fused Matrix Multiplication with Bias Addition Kernel
// Computes: C = A * (B^T) + bias, where A is [M x K], B is [N x K] (stored row-wise),
// and bias is a vector of length N. Uses shared memory tiling for improved performance.
//---------------------------------------------------------------------------
void FusedMatMulBiasKernel(const float* A,
                           const float* B,
                           const float* bias,
                           float* C,
                           int M, int N, int K) {
    #pragma omp parallel
    {
        #pragma omp for
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < N; col++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[row * K + k] * B[col * K + k];
                }
                C[row * N + col] = sum + bias[col];
            }
        }
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
//---------------------------------------------------------------------------
void FusedPoolActMaxKernel(const float* linear_output,
                           float* output,
                           int M, int N,
                           int pool_kernel_size,
                           int output_length,
                           float scale_factor) {
    #pragma omp parallel for
    for (int row = 0; row < M; row++) {
        float local_max = -FLT_MAX;
        for (int bin = 0; bin < output_length; bin++) {
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
            float gelu = 0.5f * avg * (1.0f + std::erf(avg * 0.70710678f));
            // Scale the activated output
            gelu *= scale_factor;
            local_max = std::fmaxf(local_max, gelu);
        }
        output[row] = local_max;
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

    TORCH_CHECK(x.is_cpu(), "x must be a CPU tensor");
    TORCH_CHECK(weight.is_cpu(), "weight must be a CPU tensor");
    TORCH_CHECK(bias.is_cpu(), "bias must be a CPU tensor");

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
    torch::Tensor linear_output = torch::empty({M, N}, options);

    // Launch fused matrix multiplication + bias addition kernel
    FusedMatMulBiasKernel(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        linear_output.data_ptr<float>(),
        M, N, K);

    // Determine pooling output length
    int output_length = (N + pool_kernel_size - 1) / pool_kernel_size;

    // Allocate tensor for final output (one value per batch row)
    torch::Tensor output = torch::empty({M}, options);

    // Launch fused pooling, activation, scaling, and max reduction kernel
    FusedPoolActMaxKernel(
         linear_output.data_ptr<float>(),
         output.data_ptr<float>(),
         M, N, pool_kernel_size, output_length, scale_factor);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused CPU forward (MatMul+Bias, Pool, GELU, Scale, Max Reduction)");
}