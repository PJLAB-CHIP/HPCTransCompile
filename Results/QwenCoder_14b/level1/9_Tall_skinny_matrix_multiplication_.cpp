#include <torch/extension.h>
#include <omp.h>

#define BLOCK_SIZE 16
#define ELEMENTS_PER_THREAD 4

// Function to fetch matrix elements considering transpose
inline float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

void unrolled_matmul_cpu(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         int M, int N, int K,
                         int lda, int ldb, int ldc,
                         bool transA, bool transB) {
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    #pragma omp parallel for collapse(2)
    for (int t = 0; t < numTiles; ++t) {
        int tiledK = t * BLOCK_SIZE;

        // Shared memory tiles
        float Bs[BLOCK_SIZE][BLOCK_SIZE];
        float As[ELEMENTS_PER_THREAD][BLOCK_SIZE][BLOCK_SIZE];

        // Load tile of B into shared memory with bounds check
        for (int thread_row = 0; thread_row < BLOCK_SIZE; ++thread_row) {
            for (int thread_col = 0; thread_col < BLOCK_SIZE; ++thread_col) {
                if (tiledK + thread_row < K && thread_col < N)
                    Bs[thread_row][thread_col] = get_element(B, tiledK + thread_row, thread_col, ldb, transB);
                else
                    Bs[thread_row][thread_col] = 0.0f;
            }
        }

        // Load a tile of A into shared memory. Each thread loads ELEMENTS_PER_THREAD elements
        for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
            for (int thread_row = 0; thread_row < BLOCK_SIZE; ++thread_row) {
                for (int thread_col = 0; thread_col < BLOCK_SIZE; ++thread_col) {
                    int row = e * BLOCK_SIZE + thread_row;
                    if (row < M && tiledK + thread_col < K)
                        As[e][thread_row][thread_col] = get_element(A, row, tiledK + thread_col, lda, transA);
                    else
                        As[e][thread_row][thread_col] = 0.0f;
                }
            }
        }

        // Multiply the loaded tiles
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                for (int thread_row = 0; thread_row < BLOCK_SIZE; ++thread_row) {
                    for (int thread_col = 0; thread_col < BLOCK_SIZE; ++thread_col) {
                        C[row * ldc + col] += As[e][thread_row][k] * Bs[k][thread_col];
                    }
                }
            }
        }
    }
}

torch::Tensor matmul_cpu(torch::Tensor A, torch::Tensor B) {
    if (A.device().type() != torch::kCPU || B.device().type() != torch::kCPU) {
        throw std::invalid_argument("Input tensors must be on CPU devices");
    }

    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    int lda = A.stride(0);
    int ldb = B.stride(0);
    int ldc = B.size(1);

    bool transA = false;
    bool transB = false;

    auto C = torch::empty({M, N}, A.options());

    unrolled_matmul_cpu(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K, lda, ldb, ldc, transA, transB);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cpu, "Matrix multiplication with unrolled loops optimization (CPU)");
}
