#include <torch/extension.h>
#include <omp.h>

#define TILE 32

void bmm_tiled_shared_memory_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    #pragma omp parallel
    {
        int bx = omp_get_thread_num() % (N / TILE);
        int by = omp_get_thread_num() / (N / TILE);

        __shared__ float As[TILE][TILE];
        __shared__ float Bs[TILE][TILE];

        float sum = 0.0f;

        for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
            int row = by * TILE + omp_get_thread_num() % TILE;
            int col = bx * TILE + omp_get_thread_num() / TILE;

            if (row < M && (t * TILE + omp_get_thread_num() % TILE) < K) {
                As[omp_get_thread_num() % TILE][omp_get_thread_num() / TILE] = A[(batch_size * row + omp_get_thread_num() / TILE) * K + t * TILE + omp_get_thread_num() % TILE];
            } else {
                As[omp_get_thread_num() % TILE][omp_get_thread_num() / TILE] = 0.0f;
            }

            if ((t * TILE + omp_get_thread_num() % TILE) < K && col < N) {
                Bs[omp_get_thread_num() % TILE][omp_get_thread_num() / TILE] = B[(batch_size * (t * TILE + omp_get_thread_num() % TILE) + col) * N + omp_get_thread_num() / TILE];
            } else {
                Bs[omp_get_thread_num() % TILE][omp_get_thread_num() / TILE] = 0.0f;
            }

            __syncthreads();

            for (int i = 0; i < TILE; ++i) {
                sum += As[omp_get_thread_num() % TILE][i] * Bs[i][omp_get_thread_num() % TILE];
            }

            __syncthreads();
        }

        if (row < M && col < N) {
            C[(batch_size * row + col) * N + omp_get_thread_num() / TILE] = sum;
        }
    }
}

torch::Tensor forward_bmm_shared_memory(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    bmm_tiled_shared_memory_kernel(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm_shared_memory, "Batched matrix multiplication with shared memory optimization (CUDA)");
}