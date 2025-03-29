#include <torch/extension.h>
#include <omp.h>

// Function for matrix-vector multiplication
template <typename scalar_t>
void matvec_mul_cpu(
    const scalar_t* A,
    const scalar_t* B,
    scalar_t* C,
    const int64_t M,
    const int64_t K)
{
    #pragma omp parallel for
    for (int64_t row = 0; row < M; ++row) {
        scalar_t sum = 0;
        const scalar_t* row_ptr = A + row * K;

        for (int64_t k = 0; k < K; ++k) {
            sum += row_ptr[k] * B[k];
        }

        C[row] = sum;
    }
}

torch::Tensor matvec_mul_cpu(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(!A.is_cuda(), "A must be a CPU tensor");
    TORCH_CHECK(!B.is_cuda(), "B must be a CPU tensor");
    
    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();
    
    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    
    auto B_flat = B_contig.view({-1});
    auto C = torch::zeros({M}, A.options());
    
    matvec_mul_cpu<scalar_t>(
        A_contig.data_ptr<scalar_t>(),
        B_flat.data_ptr<scalar_t>(),
        C.data_ptr<scalar_t>(),
        M,
        K
    );
    
    return C.view({M, 1});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cpu, "Matrix-Vector Multiplication (CPU)");
}
