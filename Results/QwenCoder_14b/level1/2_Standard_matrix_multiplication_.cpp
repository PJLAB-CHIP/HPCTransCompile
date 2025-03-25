#include <torch/extension.h>
#include <omp.h>

#define TILE_SIZE 32

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

// Custom tiled matrix multiplication function
void matmul_cpu(const float* A, const float* B, float* C, const int M, const int N, const int K) {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// Hybrid matrix multiplication: uses custom kernel for small matrices
torch::Tensor matrix_multiply_cpu(const torch::Tensor &A, const torch::Tensor &B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();

    auto options = torch::TensorOptions()
                       .dtype(A.dtype())
                       .device(A.device())
                       .requires_grad(false);
    
    torch::Tensor C = torch::empty({M, N}, options);
    float* d_C = C.data_ptr<float>();

    // Heuristic: use custom kernel for small matrices.
    if (M <= 128 && N <= 128 && K <= 128) {
        matmul_cpu(d_A, d_B, d_C, M, N, K);
    } else {
        // For larger matrices, use the custom CPU kernel as well
        matmul_cpu(d_A, d_B, d_C, M, N, K);
    }

    return C;
}

// PyTorch forward interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    return matrix_multiply_cpu(A, B);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid matrix multiplication (CPU): custom kernel for small matrices");
}