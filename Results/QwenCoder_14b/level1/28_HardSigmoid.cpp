#include <torch/extension.h>
#include <omp.h>

template<typename scalar_t, int VEC_SIZE>
void hardsigmoid_cpu(const scalar_t* __restrict__ input,
                      scalar_t* __restrict__ output,
                      size_t numel) {
    constexpr scalar_t three = 3.0;
    constexpr scalar_t sixth = 1.0/6.0;

    #pragma omp parallel for
    for (int i = 0; i < numel; i += VEC_SIZE) {
        scalar_t elements[VEC_SIZE];
        
        // Load elements
        for (int v = 0; v < VEC_SIZE && i + v < numel; v++) {
            elements[v] = input[i + v];
        }

        #pragma unroll
        for (int v = 0; v < VEC_SIZE && i + v < numel; v++) {
            scalar_t x = elements[v];
            x = fma(x, sixth, three * sixth);  // (x + 3) / 6
            x = std::max(0.0, std::min(1.0, x));  // Standard math functions
            elements[v] = x;
        }

        // Store elements
        for (int v = 0; v < VEC_SIZE && i + v < numel; v++) {
            output[i + v] = elements[v];
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(!input.is_cuda(), "Input tensor must be on CPU");
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();

    constexpr int VEC_SIZE = 4;  // Assuming float for simplicity

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_cpu", ([&] {
        hardsigmoid_cpu<scalar_t, VEC_SIZE>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardSigmoid activation forward (CPU)");
}