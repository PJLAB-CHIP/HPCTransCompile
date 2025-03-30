#include <torch/extension.h>
#include <omp.h>

#define TILE_SIZE 32

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Hybrid matrix multiplication: chooses custom kernel for small matrices, cuBLAS otherwise.
void matrix_multiply_cpu(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // Heuristic: use custom kernel for small matrices, cuBLAS otherwise.
    if (M <= 128 && N <= 128 && K <= 128) {
        // Launch custom tiled kernel
        #pragma omp parallel for
        for (int row = 0; row < M; ++row) {
            for (int col = 0; col < N; ++col) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += d_A[row * K + k] * d_B[k * N + col];
                }
                d_C[row * N + col] = sum;
            }
        }
    } else {
        // Initialize cuBLAS handle if needed
        static cublasHandle_t handle = nullptr;
        if (handle == nullptr) {
            cublasCreate(&handle);
            // Optionally, set math mode to use Tensor Cores if available
            cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Note: cuBLAS assumes column-major order. Here we use arguments in a way that allows using row-major data.
        // We swap A and B pointers so that C = A*B is computed correctly.
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,  // B's leading dimension
                    d_A, K,  // A's leading dimension
                    &beta,
                    d_C, N); // C's leading dimension
    }
}

// PyTorch forward interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int M = A.size(0);
    const int N = B.size(1);

    auto options = torch::TensorOptions()
                       .dtype(A.dtype())
                       .device(A.device())
                       .requires_grad(false);
    
    torch::Tensor C = torch::empty({M, N}, options);
    matrix_multiply_cpu(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid matrix multiplication (CPU): custom kernel for small matrices and cuBLAS for large matrices");
}