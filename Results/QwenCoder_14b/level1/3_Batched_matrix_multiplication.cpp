#include <torch/extension.h>
#include <omp.h>

#define TILE 32

void bmm_tiled_cpu(
    const float* A,
    const float* B,
    float* C,
    int batch_size,
    int M,
    int K,
    int N
) {
    #pragma omp parallel for collapse(3)
    for (int bz = 0; bz < batch_size; ++bz) {
        for (int by = 0; by < (M + TILE - 1) / TILE; ++by) {
            for (int bx = 0; bx < (N + TILE - 1) / TILE; ++bx) {
                float As[TILE][TILE];
                float Bs[TILE][TILE];

                for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
                    for (int ty = 0; ty < TILE; ++ty) {
                        for (int tx = 0; tx < TILE; ++tx) {
                            int row = by * TILE + ty;
                            int col = bx * TILE + tx;
                            int a_col = tx;
                            int b_row = ty;

                            if (row < M && (t * TILE + a_col) < K) {
                                As[ty][tx] = A[bz * M * K + row * K + (t * TILE + a_col)];
                            } else {
                                As[ty][tx] = 0.0f;
                            }

                            if ((t * TILE + b_row) < K && col < N) {
                                Bs[ty][tx] = B[bz * K * N + (t * TILE + b_row) * N + col];
                            } else {
                                Bs[ty][tx] = 0.0f;
                            }
                        }
                    }

                    float sum = 0.0f;
                    for (int i = 0; i < TILE; ++i) {
                        sum += As[ty][i] * Bs[i][tx];
                    }

                    if (row < M && col < N) {
                        C[bz * M * N + row * N + col] = sum;
                    }
                }
            }
        }
    }
}

torch::Tensor forward_bmm_cpu(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(!A.is_cuda(), "A must be a CPU tensor");
    TORCH_CHECK(!B.is_cuda(), "B must be a CPU tensor");
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

    bmm_tiled_cpu(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm_cpu, "Batched matrix multiplication with CPU implementation");
}
