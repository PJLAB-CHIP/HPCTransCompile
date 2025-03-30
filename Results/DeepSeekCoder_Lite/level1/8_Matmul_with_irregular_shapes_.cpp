#include <torch/extension.h>
#include <omp.h>

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

torch::Tensor matmul_cpu(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::empty({M, N}, A.options());

    float *A_data = A.data_ptr<float>();
    float *B_data = B.data_ptr<float>();
    float *C_data = C.data_ptr<float>();

    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A_data[i * K + k] * B_data[k * N + j];
            }
            C_data[i * N + j] = sum;
        }
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cpu, "CPU matmul implementation");
}