#include <torch/extension.h>
#include <cmath>
#include <omp.h>

// Define an inline exponential function for float
inline float my_exp(float x) {
    return expf(x);
}

inline void process_element(float x, float& result) {
    result = (x > 0.0f)
        ? x
        : 1.67326324235437728481f * (my_exp(x) - 1.0f);
    result *= 1.05070098735548049342f;
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(!input.is_cuda(), "Input tensor must be a CPU tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "Input must be float32");

    auto output = torch::empty_like(input);
    const size_t numel = input.numel();

    #pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        float result;
        process_element(input.data_ptr<float>()[i], result);
        output.data_ptr<float>()[i] = result;
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward (CPU)");
}
