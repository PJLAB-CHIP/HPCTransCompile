#include <torch/extension.h>
#include <omp.h>

#define TILE_WIDTH 16
#define MATRIX_SIZE_THRESHOLD 512  // Threshold to switch between custom kernel and cuBLAS

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Choose between custom kernel and cuBLAS based on matrix size
    if (M <= MATRIX_SIZE_THRESHOLD && N <= MATRIX_SIZE_THRESHOLD) {
        // Use custom kernel for smaller matrices
        #pragma omp parallel for
        for (int row = 0; row < M; ++row) {
            for (int col = 0; col < N; ++col) {
                float cValue = 0.0f;
                for (int k = 0; k < K; ++k) {
                    cValue += A[row * K + k] * B[k * N + col];
                }
                C[row * N + col] = cValue;
            }
        }
    } else {
        // Use cuBLAS for larger matrices
        static cublasHandle_t handle = nullptr;
        if (handle == nullptr) {
            cublasCreate(&handle);
        }
        
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                   N, M, K, &alpha, 
                   B.data_ptr<float>(), N, 
                   A.data_ptr<float>(), K, 
                   &beta, C.data_ptr<float>(), N);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Adaptive matrix multiplication (CPU)");
}