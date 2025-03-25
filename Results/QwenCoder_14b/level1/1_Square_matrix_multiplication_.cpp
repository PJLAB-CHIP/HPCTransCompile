#include <torch/extension.h>
#include <omp.h>
#include <vector>

#define BLOCK_SIZE 16
#define TILE_DIM (BLOCK_SIZE * 2)

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

void matmul_regtile_optimized_cpu(const float* A, const float* B, float* C, int N) {
    int numTiles = (N + TILE_DIM - 1) / TILE_DIM;

    #pragma omp parallel for collapse(2)
    for (int by = 0; by < numTiles; ++by) {
        for (int bx = 0; bx < numTiles; ++bx) {
            float regC00 = 0.0f, regC01 = 0.0f, regC10 = 0.0f, regC11 = 0.0f;

            for (int t = 0; t < numTiles; ++t) {
                for (int r = 0; r < TILE_DIM; ++r) {
                    for (int c = 0; c < TILE_DIM; ++c) {
                        int globalA_r = by * TILE_DIM + r;
                        int globalA_c = t * TILE_DIM + c;
                        float a0 = (globalA_r < N && globalA_c < N) ? A[globalA_r * N + globalA_c] : 0.0f;

                        int globalB_r = t * TILE_DIM + r;
                        int globalB_c = bx * TILE_DIM + c;
                        float b0 = (globalB_r < N && globalB_c < N) ? B[globalB_r * N + globalB_c] : 0.0f;

                        regC00 += a0 * b0;
                    }
                }
            }

            int row = by * TILE_DIM;
            int col = bx * TILE_DIM;
            if (row < N && col < N) C[row * N + col] = regC00;
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    CHECK_FLOAT(A); CHECK_FLOAT(B);
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must be square and equal size");

    int64_t N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    matmul_regtile_optimized_cpu(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 2x2 tiling MM with sync reduction");
}