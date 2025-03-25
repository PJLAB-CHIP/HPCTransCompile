#include <torch/extension.h>
#include <omp.h>

#define TILE_SIZE 32

void coalesced_memory_access_upper_triangular_cpu(const float* A, const float* B, float* C, const int N) {
    #pragma omp parallel for collapse(2)
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

torch::Tensor coalesced_memory_access_upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    coalesced_memory_access_upper_triangular_cpu(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &coalesced_memory_access_upper_triangular_matmul, "Coalesced memory access upper triangular matrix multiplication");
}