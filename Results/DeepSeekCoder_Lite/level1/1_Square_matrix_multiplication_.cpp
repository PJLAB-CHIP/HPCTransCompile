#include <torch/extension.h>
#include <omp.h>

#define BLOCK_SIZE 16
#define TILE_DIM (BLOCK_SIZE * 2)
#define PADDING 1  // Avoid shared memory bank conflicts

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

void matmul_regtile_optimized_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    int numTiles = (N + TILE_DIM - 1) / TILE_DIM;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int by = 0; by < numTiles; ++by) {
        for (int bx = 0; bx < numTiles; ++bx) {
            float regC00 = 0.0f, regC01 = 0.0f, regC10 = 0.0f, regC11 = 0.0f;

            for (int ty = 0; ty < BLOCK_SIZE; ++ty) {
                for (int tx = 0; tx < BLOCK_SIZE; ++tx) {
                    int row = by * TILE_DIM + ty * 2;
                    int col = bx * TILE_DIM + tx * 2;

                    for (int t = 0; t < numTiles; ++t) {
                        int globalA_r = by * TILE_DIM + ty * 2 + 0;
                        int globalA_c = t * TILE_DIM + tx * 2 + 0;
                        float a0 = (globalA_r < N && globalA_c < N) ? A[globalA_r * N + globalA_c] : 0.0f;

                        int globalB_r = t * TILE_DIM + ty * 2 + 0;
                        int globalB_c = bx * TILE_DIM + tx * 2 + 0;
                        float b0 = (globalB_r < N && globalB_c < N) ? B[globalB_r * N + globalB_c] : 0.0f;
                        regC00 += a0 * b0;

                        int globalA_r_1 = by * TILE_DIM + ty * 2 + 1;
                        int globalA_c_1 = t * TILE_DIM + tx * 2 + 0;
                        float a1 = (globalA_r_1 < N && globalA_c_1 < N) ? A[globalA_r_1 * N + globalA_c_1] : 0.0f;

                        int globalB_r_1 = t * TILE_DIM + ty * 2 + 1;
                        int globalB_c_1 = bx * TILE_DIM + tx * 2 + 0;
                        float b1 = (globalB_r_1 < N && globalB_c_1 < N) ? B[globalB_r_1 * N + globalB_c_1] : 0.0f;
                        regC01 += a1 * b0;
                        regC10 += a0 * b1;
                        regC11 += a1 * b1;
                    }
                }
            }

            if (row < N && col < N) C[row * N + col] = regC00;
            if (row < N && col + 1 < N) C[row * N + col + 1] = regC01;
            if (row + 1 < N && col < N) C[(row + 1) * N + col] = regC10;
            if (row + 1 < N && col + 1 < N) C[(row + 1) * N + col + 1] = regC11;
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    CHECK_FLOAT(A); CHECK_FLOAT(B);
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must be square and equal size");

    int64_t N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    matmul_regtile_optimized_kernel(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 2x2 tiling MM with sync reduction");
}