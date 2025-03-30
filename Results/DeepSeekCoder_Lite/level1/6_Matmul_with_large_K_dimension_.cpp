#include <torch/extension.h>
#include <omp.h>

#define TILE_WIDTH 32

template <typename scalar_t>
void matmul_double_buffered(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
                             scalar_t* __restrict__ C, int M, int K, int N) {
    #pragma omp parallel
    {
        #pragma omp for
        for (int row = 0; row < M; ++row) {
            for (int col = 0; col < N; ++col) {
                scalar_t accum = 0;
                for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
                    scalar_t sA = 0, sB = 0;
                    if (t * TILE_WIDTH + threadIdx.x < K && row < M && col < N) {
                        sA = A[row * K + t * TILE_WIDTH + threadIdx.x];
                    }
                    if (t * TILE_WIDTH + threadIdx.y < K && row < M && col < N) {
                        sB = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
                    }
                    accum += sA * sB;
                }
                if (row < M && col < N) {
                    C[row * N + col] = accum;
                }
            }
        }
    }
}

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions must match");

    auto C = torch::empty({M, N}, A.options());

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_double_buffered", ([&] {
        matmul_double_buffered<scalar_t>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Double buffered matrix multiplication");
}