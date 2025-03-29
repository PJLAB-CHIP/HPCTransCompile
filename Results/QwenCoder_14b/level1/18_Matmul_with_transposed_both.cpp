#include <torch/extension.h>
#include <omp.h>

template<int BLOCK_SIZE = 32>
struct LocalMemoryTile {
    template <typename scalar_t>
    static void loadA(
        scalar_t (&tileA)[BLOCK_SIZE][BLOCK_SIZE],
        const scalar_t* A,
        const int row,
        const int tile_idx,
        const int M,
        const int K) {
        const int k_index = tile_idx * BLOCK_SIZE;
        for (int y = 0; y < BLOCK_SIZE; ++y) {
            if (k_index + y < K && row < M) {
                tileA[y][0] = A[(k_index + y) * M + row];
            } else {
                tileA[y][0] = 0.0;
            }
        }
    }

    template <typename scalar_t>
    static void loadB(
        scalar_t (&tileB)[BLOCK_SIZE][BLOCK_SIZE],
        const scalar_t* B,
        const int col,
        const int tile_idx,
        const int N,
        const int K) {
        const int k_index = tile_idx * BLOCK_SIZE;
        for (int x = 0; x < BLOCK_SIZE; ++x) {
            if (k_index + x < K && col < N) {
                tileB[0][x] = B[col * K + (k_index + x)];
            } else {
                tileB[0][x] = 0.0;
            }
        }
    }

    template <typename scalar_t>
    static scalar_t computeTileProduct(
        const scalar_t (&tileA)[BLOCK_SIZE][BLOCK_SIZE],
        const scalar_t (&tileB)[BLOCK_SIZE][BLOCK_SIZE]) {
        scalar_t sum = 0;
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += tileA[k][0] * tileB[0][k];
        }
        return sum;
    }
};

template <typename scalar_t, int BLOCK_SIZE = 32>
void matmul_transpose_cpu(
    const scalar_t* A,
    const scalar_t* B,
    scalar_t* C,
    const int M,
    const int N,
    const int K) {
    
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            scalar_t sum = 0;
            
            #pragma unroll 4
            for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
                scalar_t tileA[BLOCK_SIZE][BLOCK_SIZE];
                scalar_t tileB[BLOCK_SIZE][BLOCK_SIZE];
                
                LocalMemoryTile<BLOCK_SIZE>::loadA(tileA, A, row, t, M, K);
                LocalMemoryTile<BLOCK_SIZE>::loadB(tileB, B, col, t, N, K);
                
                sum += LocalMemoryTile<BLOCK_SIZE>::computeTileProduct(tileA, tileB);
            }

            C[row * N + col] = sum;
        }
    }
}

torch::Tensor matmul_transpose_cpu(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    constexpr int BLOCK_SIZE = 32;
    matmul_transpose_cpu<scalar_t, BLOCK_SIZE>(
        A.data_ptr<scalar_t>(),
        B.data_ptr<scalar_t>(),
        C.data_ptr<scalar_t>(),
        M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cpu, "Optimized matrix multiplication with transpose (CPU)");
}
