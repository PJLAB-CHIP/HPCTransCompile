#include <torch/extension.h>
#include <omp.h>

void hybrid_diag_matmul_cpu(
    const float* A,
    const float* B,
    float* C,
    const int64_t N,
    const int64_t M,
    const bool use_vectorized
) {
    if (use_vectorized) {
        // Vectorized approach for large matrices where M is divisible by 4
        #pragma omp parallel for
        for (int idx = 0; idx < N * M / 4; ++idx) {
            const int base_idx = idx * 4;
            const int row = base_idx / M;
            const float a_val = A[row];
            
            const float b_val[4] = {B[base_idx], B[base_idx + 1], B[base_idx + 2], B[base_idx + 3]};
            const float c_val[4] = {a_val * b_val[0], a_val * b_val[1], a_val * b_val[2], a_val * b_val[3]};
            
            C[base_idx] = c_val[0];
            C[base_idx + 1] = c_val[1];
            C[base_idx + 2] = c_val[2];
            C[base_idx + 3] = c_val[3];
        }
    } else {
        // Row-based approach for smaller matrices or when M is not divisible by 4
        #pragma omp parallel for
        for (int row = 0; row < N; ++row) {
            float a_val = A[row];
            const int main_end = (M / 4) * 4;
            
            // Main loop with coalesced access
            for (int j = 0; j < main_end; j += 4) {
                int idx = row * M + j;
                C[idx] = a_val * B[idx];
                C[idx + 1] = a_val * B[idx + 1];
                C[idx + 2] = a_val * B[idx + 2];
                C[idx + 3] = a_val * B[idx + 3];
            }
            
            // Handle remaining elements
            for (int j = main_end; j < M; ++j) {
                int idx = row * M + j;
                C[idx] = a_val * B[idx];
            }
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch");

    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);
    auto C = torch::empty({N, M}, B.options());

    // Choose approach based on matrix size and alignment
    bool use_vectorized = (M >= 512) && (M % 4 == 0);
    
    hybrid_diag_matmul_cpu(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        N, M, use_vectorized);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid diagonal matrix multiplication");
}
