#include <torch/extension.h>
#include <omp.h>

#define TILE_DIM 32

// This function performs 3D tensor-matrix multiplication for A (N x M x K) and B (K x L).
// The output is a tensor of shape (N x M x L) computed by flattening the first two dimensions of A into (N*M) x K.

template <typename scalar_t>
void unrolled_tiled_cpu_function(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    int N, int M, int K, int L) {

    #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            int global_row = n * M + m;  // Flatten the (N, M) output indices into one
            scalar_t sum = 0;

            int numTiles = (K + TILE_DIM - 1) / TILE_DIM;

            // Loop over tiles in the K dimension; unroll for reduced loop overhead when possible
            #pragma unroll
            for (int t = 0; t < numTiles; ++t) {
                int A_col = t * TILE_DIM;
                int B_row = t * TILE_DIM;

                // Compute partial dot-product for the current tile; unroll inner loop
                #pragma unroll
                for (int i = 0; i < TILE_DIM; ++i) {
                    if (A_col + i < K && B_row + i < K) {
                        sum += A[global_row * K + A_col + i] * B[B_row * L + i];
                    }
                }
            }

            // Write the output
            output[global_row * L] = sum;
        }
    }
}

// CPU forward function to launch the unrolled_tiled_cpu_function
void module_fn_cpu_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);

    unrolled_tiled_cpu_function<scalar_t>(
        A.data_ptr<scalar_t>(),
        B.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        N, M, K, L);
}

// Macros for input checking
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// C++ interface that exposes the forward function to Pybind11
torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);

    const int N = A.size(0);
    const int M = A.size(1);
    const int L = B.size(1);

    auto output = torch::zeros({N, M, L}, A.options());
    module_fn_cpu_forward(A, B, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Unrolled tiled tensor-matrix multiplication (CPU)");
}
