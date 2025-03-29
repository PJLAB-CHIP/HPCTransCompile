#include <torch/extension.h>
#include <omp.h>

void vectorizedMultiplyCPU(const float* A, float* C, float s, int64_t size)
{
    #pragma omp parallel for
    for (int64_t idx = 0; idx < size; idx += 4) {
        int64_t idx4 = idx;
        
        // Process 4 elements at a time
        if (idx4 + 3 < size) {
            C[idx4] = A[idx4] * s;
            C[idx4 + 1] = A[idx4 + 1] * s;
            C[idx4 + 2] = A[idx4 + 2] * s;
            C[idx4 + 3] = A[idx4 + 3] * s;
        }
        // Handle remaining elements
        else {
            for (int64_t i = 0; i < 4 && idx4 + i < size; i++) {
                C[idx4 + i] = A[idx4 + i] * s;
            }
        }
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(!A.is_cuda(), "Input tensor A must be a CPU tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");
    
    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    
    vectorizedMultiplyCPU(A.data_ptr<float>(), C.data_ptr<float>(), s, size);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized matrix-scalar multiplication CPU");
}