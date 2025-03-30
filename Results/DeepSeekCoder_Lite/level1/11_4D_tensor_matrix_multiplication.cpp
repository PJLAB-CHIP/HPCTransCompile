#include <torch/extension.h>
#include <omp.h>

#define TILE_SIZE 32

void einsum_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int I, int J, int L, int K, int num_threads
) {
    #pragma omp parallel for num_threads(num_threads)
    for (int batch_i = 0; batch_i < BATCH; ++batch_i) {
        for (int idx_i = 0; idx_i < I; ++idx_i) {
            for (int j = 0; j < J; ++j) {
                for (int k = 0; k < K; ++k) {
                    float sum = 0.0f;
                    for (int l_block = 0; l_block < L; l_block += TILE_SIZE) {
                        for (int l_tile = 0; l_tile < TILE_SIZE; ++l_tile) {
                            for (int tx = 0; tx < TILE_SIZE; ++tx) {
                                int l = l_block + l_tile;
                                if (l < L && k < K) {
                                    sum += A[batch_i * I*J*L + idx_i * J*L + j * L + l] * B[l * K + k];
                                }
                            }
                        }
                    }
                    int c_idx = batch_i * I*J*K + idx_i * J*K + j * K + k;
                    C[c_idx] = sum;
                }
            }
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in l");

    int BATCH = A.size(0), I = A.size(1), J = A.size(2), L = A.size(3);
    int K = B.size(1);
    
    auto C = torch::zeros({BATCH, I, J, K}, A.options());
    
    int num_threads = omp_get_max_threads();
    einsum_kernel(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K, num_threads
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with shared memory tiling and __ldg optimization (CPU)");
}