#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <omp.h>

template <typename scalar_t>
__forceinline__ scalar_t compute_softplus(const scalar_t x) {
    if (x > static_cast<scalar_t>(20.0)) {
        return x;
    } else if (x < static_cast<scalar_t>(-20.0)) {
        return exp(x);
    }
    return log1p(exp(x));
}

torch::Tensor softplus_cpu_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();

    #pragma omp parallel for
    for (int idx = 0; idx < size; ++idx) {
        const scalar_t x = input[idx];
        output[idx] = compute_softplus(x);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cpu_forward, "Softplus forward (CPU)");
}