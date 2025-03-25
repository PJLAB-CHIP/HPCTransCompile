#include <torch/extension.h>
#include <cmath>
#include <omp.h>

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

// Fused Matrix Multiplication with Bias Addition
void FusedMatMulBiasCPU(const float* A, const float* B, const float* bias, float* C, int M, int N, int K) {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[col * K + k];
            }
            C[row * N + col] = sum + bias[col];
        }
    }
}

// Fused Pooling, Activation, Scaling and Max Reduction
void FusedPoolActMaxCPU(const float* linear_output, float* output, int M, int N, int pool_kernel_size, int output_length, float scale_factor) {
    #pragma omp parallel for
    for (int row = 0; row < M; ++row) {
        float local_max = -std::numeric_limits<float>::max();
        for (int bin = 0; bin < output_length; ++bin) {
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
            float gelu = 0.5f * avg * (1.0f + std::erff(avg * 0.70710678f));
            // Scale the activated output
            gelu *= scale_factor;
            local_max = std::max(local_max, gelu);
        }
        output[row] = local_max;
    }
}

// Forward function that chains the fused operations
torch::Tensor forward(
    torch::Tensor x,
    int pool_kernel_size,
    float scale_factor,
    torch::Tensor weight,
    torch::Tensor bias) {

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

    // Perform fused matrix multiplication + bias addition
    FusedMatMulBiasCPU(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), linear_output.data_ptr<float>(), M, N, K);

    // Determine pooling output length
    int output_length = (N + pool_kernel_size - 1) / pool_kernel_size;

    // Allocate tensor for final output (one value per batch row)
    auto output = torch::empty({M}, options);

    // Perform fused pooling, activation, scaling, and max reduction
    FusedPoolActMaxCPU(linear_output.data_ptr<float>(), output.data_ptr<float>(), M, N, pool_kernel_size, output_length, scale_factor);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused CPU forward (MatMul+Bias, Pool, GELU, Scale, Max Reduction)");
}