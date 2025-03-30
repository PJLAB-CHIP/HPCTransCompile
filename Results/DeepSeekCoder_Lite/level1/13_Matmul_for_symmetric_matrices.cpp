#include <torch/extension.h>
#include <omp.h>
#include <vector>

#define BLOCK_SIZE 64
#define THREAD_TILE 4
#define MAX_MATRIX_DIM 8192

// Constant memory for matrix dimensions and number of tiles
__constant__ int d_N;
__constant__ int d_num_tiles;

// This function performs the exact same operation as the CUDA kernel code.
// It uses OpenMP for multi-threading to parallelize the computation.
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D matrices");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same dimensions");
    TORCH_CHECK(A.size(0) <= MAX_MATRIX_DIM, "Matrix dimension exceeds maximum supported size");

    int N = A.size(0);
    int num_tiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMemcpyToSymbol(d_N, &N, sizeof(int));
    cudaMemcpyToSymbol(d_num_tiles, &num_tiles, sizeof(int));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Launch configuration: blockDim = (BLOCK_SIZE/THREAD_TILE, BLOCK_SIZE/THREAD_TILE) = (16, 16)
    dim3 threads(BLOCK_SIZE / THREAD_TILE, BLOCK_SIZE / THREAD_TILE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel
    vec_ldg_aligned_matmul<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());

    return C;
}

// C++ interface using Pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized 128-bit Aligned Matrix Multiplication with __ldg (CUDA)");
}