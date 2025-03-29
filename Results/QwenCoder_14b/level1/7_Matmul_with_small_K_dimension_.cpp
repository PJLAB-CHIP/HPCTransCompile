#include <torch/extension.h>
#include <omp.h>

#define TILE_WIDTH 16
#define MATRIX_SIZE_THRESHOLD 512  // Threshold to switch between custom kernel and standard matrix multiplication

// Custom CPU kernel for smaller matrices
void MatmulKernel(const float* A, const float* B, float* C, int M, int K, int N) {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float cValue = 0.0f;
            for (int k = 0; k < K; k++) {
                cValue += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = cValue;
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(!A.is_cuda(), "A must be a CPU tensor");
    TORCH_CHECK(!B.is_cuda(), "B must be a CPU tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Choose between custom kernel and standard matrix multiplication based on matrix size
    if (M <= MATRIX_SIZE_THRESHOLD && N <= MATRIX_SIZE_THRESHOLD) {
        // Use custom kernel for smaller matrices
        MatmulKernel(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    } else {
        // Use standard matrix multiplication for larger matrices
        #pragma omp parallel for collapse(2)
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < N; col++) {
                float cValue = 0.0f;
                for (int k = 0; k < K; k++) {
                    cValue += A[row * K + k] * B[k * N + col];
                }
                C[row * N + col] = cValue;
            }
        }
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive matrix multiplication (CPU)");
}
