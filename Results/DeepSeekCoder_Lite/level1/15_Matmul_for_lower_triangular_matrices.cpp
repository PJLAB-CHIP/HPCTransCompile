#include <torch/extension.h>
#include <omp.h>

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Parallelize the computation
    #pragma omp parallel for
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            if (col <= row) {
                float sum = 0.0f;
                for (int k = col; k <= row; ++k) {
                    sum += A[row * N + k].item<float>() * B[k * N + col].item<float>();
                }
                C[row * N + col] = sum;
            } else {
                C[row * N + col] = 0.0f;
            }
        }
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Efficient triangular matrix multiplication (CPU)");
}