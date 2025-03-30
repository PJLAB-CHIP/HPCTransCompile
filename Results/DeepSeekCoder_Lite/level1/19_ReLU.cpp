#include <torch/extension.h>
#include <omp.h>

// C++ CPU implementation for ReLU activation
template <typename scalar_t>
void relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    #pragma omp parallel for
    for (int idx = 0; idx < size; ++idx) {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel", ([&] {
        relu_kernel<scalar_t>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward (CPU)");
}