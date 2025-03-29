#include <torch/extension.h>
#include <cmath>
#include <omp.h>

const int THREADS = 256;
const int ELEMENTS_PER_THREAD = 4;

template <typename scalar_t>
void sigmoid_cpu(const scalar_t* __restrict__ input,
                 scalar_t* __restrict__ output,
                 const int64_t size) {
    #pragma omp parallel for num_threads(THREADS)
    for (int64_t i = 0; i < size; i++) {
        float val = -static_cast<float>(input[i]);
        float exp_val = expf(val);
        float r = 1.0f / (1.0f + exp_val);
        output[i] = static_cast<scalar_t>(r);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_cpu", [&] {
        const auto* input_data = input.data_ptr<scalar_t>();
        auto* output_data = output.data_ptr<scalar_t>();
        
        sigmoid_cpu<scalar_t>(input_data, output_data, size);
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sigmoid forward (CPU)");
}
