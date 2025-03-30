#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

const int ELEMENTS_PER_THREAD = 4;
const int SHARED_MEM_SIZE = ELEMENTS_PER_THREAD * 256; // Assuming THREADS = 256

template <typename scalar_t>
void sigmoid_kernel(const scalar_t* input, scalar_t* output, const int64_t size) {
    #pragma omp parallel for
    for (int64_t i = 0; i < size; i += ELEMENTS_PER_THREAD) {
        for (int j = 0; j < ELEMENTS_PER_THREAD; ++j) {
            if (i + j < size) {
                scalar_t val = input[i + j];
                output[i + j] = static_cast<scalar_t>(1.0 / (1.0 + exp(-val)));
            }
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", [&] {
        const auto* input_data = input.data_ptr<scalar_t>();
        auto* output_data = output.data_ptr<scalar_t>();
        
        sigmoid_kernel<scalar_t>(input_data, output_data, size);
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sigmoid forward (CPU)");
}