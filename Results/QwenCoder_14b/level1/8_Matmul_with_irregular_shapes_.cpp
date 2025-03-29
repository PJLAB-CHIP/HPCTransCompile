#include <torch/extension.h>
#include <Eigen/Dense>
#include <omp.h>

#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

torch::Tensor matmul_cpu(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::empty({M, N}, A.options());

    Eigen::Map<Eigen::MatrixXf> eigen_A(A.data_ptr<float>(), M, K);
    Eigen::Map<Eigen::MatrixXf> eigen_B(B.data_ptr<float>(), K, N);
    Eigen::Map<Eigen::MatrixXf> eigen_C(C.data_ptr<float>(), M, N);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            eigen_C(i, j) = 0.0f;
            for (int k = 0; k < K; ++k) {
                eigen_C(i, j) += eigen_A(i, k) * eigen_B(k, j);
            }
        }
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cpu, "CPU-based matrix multiplication with OpenMP");
}