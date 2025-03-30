#include <torch/extension.h>
#include <omp.h>

void vectorizedMultiplyKernel(const float* __restrict__ A,
                              float* __restrict__ C,
                              float s,
                              int64_t size)
{
    #pragma omp parallel for
    for (int idx = 0; idx < size; ++idx) {
        int idx4 = idx * 4;
        if (idx4 + 3 < size) {
            float4 a4;
            float4* a4_ptr = (float4*)(&A[idx4]);
            float4* c4_ptr = (float4*)(&C[idx4]);
            
            a4.x = A[idx4] * s;
            a4.y = A[idx4 + 1] * s;
            a4.z = A[idx4 + 2] * s;
            a4.w = A[idx4 + 3] * s;
            
            c4_ptr->x = a4.x;
            c4_ptr->y = a4.y;
            c4_ptr->z = a4.z;
            c4_ptr->w = a4.w;
        } else {
            int base = idx4;
            for (int i = 0; i < 4 && base + i < size; i++) {
                C[base + i] = A[base + i] * s;
            }
        }
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");
    
    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    
    vectorizedMultiplyKernel(A.data_ptr<float>(), C.data_ptr<float>(), s, size);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized matrix-scalar multiplication kernel");
}