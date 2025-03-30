#include <torch/extension.h>
#include <omp.h>

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch");

    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);
    auto C = torch::empty({N, M}, B.options());

    // Choose approach based on matrix size and alignment
    bool use_vectorized = (M >= 512) && (M % 4 == 0);

    #pragma omp parallel for
    for (int64_t row = 0; row < N; ++row) {
        float a_val = A[row].item<float>();
        for (int64_t j = 0; j < M; ++j) {
            C[row * M + j] = a_val * B[row * M + j].item<float>();
        }
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid diagonal matrix multiplication");
}