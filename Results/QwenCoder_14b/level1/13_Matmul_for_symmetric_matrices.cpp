#include <torch/extension.h>
#include <omp.h>

#define BLOCK_SIZE 64
#define THREAD_TILE 4
#define MAX_MATRIX_DIM 8192

void vec_cpu_matmul(const float* A, const float* B, float* C, int N) {
    // Parallelize the outer loop using OpenMP
    #pragma omp parallel for collapse(2)
    for (int by = 0; by < N / BLOCK_SIZE; ++by) {
        for (int bx = 0; bx < N / BLOCK_SIZE; ++bx) {
            // Compute the starting global indices for the 4x4 output computed by this thread
            int rowStart = by * BLOCK_SIZE;
            int colStart = bx * BLOCK_SIZE;

            // Registers to accumulate a 4x4 sub-tile
            float regC[THREAD_TILE][THREAD_TILE] = { {0.f, 0.f, 0.f, 0.f},
                                                       {0.f, 0.f, 0.f, 0.f},
                                                       {0.f, 0.f, 0.f, 0.f},
                                                       {0.f, 0.f, 0.f, 0.f} };

            // Loop over tiles in the k-dimension
            for (int t = 0; t < N / BLOCK_SIZE; ++t) {
                // Multiply the loaded tiles; each thread computes its 4x4 sub-block.
                int a_sub_row = 0; // starting row in A for this block
                int b_sub_col = 0;   // starting col in B for this block

                #pragma unroll
                for (int k = 0; k < BLOCK_SIZE; k++) {
                    for (int i = 0; i < THREAD_TILE; ++i) {
                        for (int j = 0; j < THREAD_TILE; ++j) {
                            regC[i][j] += A[(rowStart + i) * N + (t * BLOCK_SIZE + k)] * B[(t * BLOCK_SIZE + k) * N + (colStart + j)];
                        }
                    }
                }
            }

            // Write the 4x4 sub-tile from registers back to global memory C
            for (int i = 0; i < THREAD_TILE; i++) {
                int global_row = rowStart + i;
                if (global_row < N) {
                    int global_col = colStart;
                    if (global_col + 3 < N) {
                        float4 out_val;
                        out_val.x = regC[i][0];
                        out_val.y = regC[i][1];
                        out_val.z = regC[i][2];
                        out_val.w = regC[i][3];
                        float4* C_vec_ptr = reinterpret_cast<float4*>(C);
                        int index = global_row * N + global_col;
                        C_vec_ptr[index / 4] = out_val;
                    } else {
                        for (int j = 0; j < THREAD_TILE; j++) {
                            int global_col_j = global_col + j;
                            if (global_col_j < N)
                                C[global_row * N + global_col_j] = regC[i][j];
                        }
                    }
                }
            }
        }
    }
}

// C++ interface using Pybind11

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cpu(), "A must be a CPU tensor");
    TORCH_CHECK(B.device().is_cpu(), "B must be a CPU tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D matrices");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same dimensions");
    TORCH_CHECK(A.size(0) <= MAX_MATRIX_DIM, "Matrix dimension exceeds maximum supported size");

    int N = A.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto C = torch::zeros({N, N}, options);

    vec_cpu_matmul(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized 128-bit Aligned Matrix Multiplication (CPU)");
}