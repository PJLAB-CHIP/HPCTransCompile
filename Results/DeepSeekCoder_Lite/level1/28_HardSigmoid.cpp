#include <torch/extension.h>
#include <omp.h>
#include <cmath>
#include <vector>

template<typename scalar_t, int VEC_SIZE>
void hardsigmoid_kernel_cpu(const scalar_t* __restrict__ input,
                            scalar_t* __restrict__ output,
                            size_t numel) {
    constexpr scalar_t three = 3.0;
    constexpr scalar_t sixth = 1.0 / 6.0;

    for (size_t i = 0; i < numel; i++) {
        scalar_t x = input[i];
        x = (x + three) * sixth;  // (x + 3) / 6
        x = std::fmax(0.0, std::fmin(1.0, x));  // Built-in fast math functions
        output[i] = x;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();

    constexpr int VEC_SIZE = sizeof(float) / sizeof(float);  // 4 for float, 2 for double

    #pragma omp parallel for
    for (size_t i = 0; i < numel; i += VEC_SIZE) {
        hardsigmoid_kernel_cpu<float, VEC_SIZE>(input.data_ptr<float>(), output.data_ptr<float>(), numel);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardSigmoid activation forward (CPU)");
}