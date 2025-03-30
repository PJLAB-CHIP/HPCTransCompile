#include <torch/extension.h>
#include <omp.h>
#include <stdexcept>

// Define tile and block dimensions
#define BLOCK_M 16        // Number of C-rows computed per block
#define BLOCK_N 32        // Number of C-columns computed per block (each thread computes 2 outputs)
#define TILE 16           // Tile width for the K dimension

// The forward function exposed via PyBind11
// Inputs:
//   A: Tensor of shape (K, M) [CPU, float32]
//   B: Tensor of shape (K, N) [CPU, float32]
// Returns:
//   C: Tensor of shape (M, N) computed as A.T * B.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() == false, "Input A must be a CPU tensor");
    TORCH_CHECK(B.is_cuda() == false, "Input B must be a CPU tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(torch::kCPU).dtype(torch::kFloat32));

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Parallelize the computation using OpenMP
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; col += 2) {
            float out0 = 0.0f, out1 = 0.0f;

            for (int k = 0; k < K; ++k) {
                float a_val = A_ptr[k * M + row];
                out0 += a_val * B_ptr[k * N + col];
                out1 += a_val * B_ptr[k * N + col + 1];
            }

            C_ptr[row * N + col] = out0;
            if (col + 1 < N) {
                C_ptr[row * N + col + 1] = out1;
            }
        }
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using tiled kernel with double output per thread (CPU)");
}