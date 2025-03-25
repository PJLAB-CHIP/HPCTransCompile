#include <torch/extension.h>
#include <omp.h>

// Function to clamp the value to a specified minimum
template <typename scalar_t>
scalar_t apply_clamp(scalar_t x, float min_val) {
    scalar_t min_cast = static_cast<scalar_t>(min_val);
    return (x < min_cast) ? min_cast : x;
}

// Function to perform division
template <typename scalar_t>
scalar_t apply_divide(scalar_t x, float divisor) {
    return x / static_cast<scalar_t>(divisor);
}

// Modular function combining clamp and divide operations
template <typename scalar_t>
scalar_t clamp_and_divide(scalar_t x, float min_val, float divisor) {
    return apply_divide(apply_clamp(x, min_val), divisor);
}

// CPU function applying the clamp and divide operation on each element
template <typename scalar_t>
void clamp_and_divide_cpu(scalar_t* output, int64_t numel, float min_val, float divisor) {
    #pragma omp parallel for
    for (int i = 0; i < numel; i++) {
        output[i] = clamp_and_divide(output[i], min_val, divisor);
    }
}

// Forward function performing 3D transposed convolution, then applying the CPU function
torch::Tensor forward(torch::Tensor input,
                      int stride,
                      int padding,
                      float min_val,
                      float divisor,
                      torch::Tensor weight,
                      torch::Tensor bias) {
    // Execute 3D transposed convolution via PyTorch
    auto output = torch::conv_transpose3d(input, weight, bias, stride, padding);

    AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "clamp_and_divide", ([&] {
        clamp_and_divide_cpu<scalar_t>(
            output.data_ptr<scalar_t>(),
            output.numel(),
            min_val,
            divisor
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Transposed convolution with clamp and divide (CPU)");
}