#include <torch/extension.h>
#include <omp.h>

// CPU implementation of the warp_matmul_optimized_v2_kernel
void warp_matmul_optimized_v2_kernel_cpu(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int M, int N, int K) {
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

// Forward function callable from PyTorch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have the same K dimension");
    TORCH_CHECK(!A.is_cuda() && !B.is_cuda(), "Inputs must be on CPU");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    warp_matmul_optimized_v2_kernel_cpu(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-level optimized matrix multiplication with transposed B (CPU)");
}