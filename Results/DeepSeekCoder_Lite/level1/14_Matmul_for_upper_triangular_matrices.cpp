#include <torch/extension.h>
#include <omp.h>

#define TILE_SIZE 32

void coalesced_memory_access_upper_triangular_kernel(const float* __restrict__ A,
                                                      const float* __restrict__ B,
                                                      float* __restrict__ C,
                                                      const int N) {
    #pragma omp parallel for
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            if (row <= col) {
                float sum = 0.0f;
                for (int t = row / TILE_SIZE * TILE_SIZE; t <= col; t += TILE_SIZE) {
                    for (int k = 0; k < TILE_SIZE; ++k) {
                        int global_k = t + k;
                        if (global_k >= row && global_k <= col && global_k < N) {
                            sum += A[row * N + global_k] * B[global_k * N + col];
                        }
                    }
                }
                C[row * N + col] = sum;
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &coalesced_memory_access_upper_triangular_kernel, "Coalesced memory access upper triangular matrix multiplication");
}