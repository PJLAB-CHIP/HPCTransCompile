#include <torch/extension.h>
#include <omp.h>
#include <stdexcept>

// Define tile and block dimensions
#define BLOCK_M 16        // Number of C-rows computed per block
#define BLOCK_N 32        // Number of C-columns computed per block (each thread computes 2 outputs)
#define TILE 16           // Tile width for the K dimension

// Function to compute C = A.T * B, where A is (K, M), B is (K, N) and C is (M, N).
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(!A.is_cuda(), "Input A must be a CPU tensor");
    TORCH_CHECK(!B.is_cuda(), "Input B must be a CPU tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Parallelize the computation using OpenMP
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; col += 2) {
            float out0 = 0.0f, out1 = 0.0f;

            for (int k = 0; k < K; ++k) {
                out0 += A[k * M + row] * B[k * N + col];
                if (col + 1 < N) {
                    out1 += A[k * M + row] * B[k * N + col + 1];
                }
            }

            C[row * N + col] = out0;
            if (col + 1 < N) {
                C[row * N + col + 1] = out1;
            }
        }
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using tiled kernel with double output per thread (CPU)");
}