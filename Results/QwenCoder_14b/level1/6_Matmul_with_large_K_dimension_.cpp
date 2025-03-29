#include <torch/extension.h>
#include <omp.h>

#define TILE_WIDTH 32

template <typename scalar_t>
void matmul_double_buffered_cpu(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
                                scalar_t* __restrict__ C, int M, int K, int N) {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            scalar_t accum = 0;
            for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
                for (int i = 0; i < TILE_WIDTH; ++i) {
                    if (row < M && t * TILE_WIDTH + i < K && col < N && t * TILE_WIDTH + i < K) {
                        accum += A[row * K + t * TILE_WIDTH + i] * B[(t * TILE_WIDTH + i) * N + col];
                    }
                }
            }
            C[row * N + col] = accum;
        }
    }
}

torch::Tensor module_fn_cpu(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(!A.is_cuda(), "Input tensor A must be a CPU tensor");
    TORCH_CHECK(!B.is_cuda(), "Input tensor B must be a CPU tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions must match");

    auto C = torch::empty({M, N}, A.options());

    matmul_double_buffered_cpu<scalar_t>(
        A.data_ptr<scalar_t>(),
        B.data_ptr<scalar_t>(),
        C.data_ptr<scalar_t>(),
        M, K, N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_cpu, "Double buffered matrix multiplication on CPU");
}
