#include <torch/extension.h>
#include <vector>
#include <omp.h>

// Device function to clamp the value to a specified minimum
template <typename scalar_t>
__device__ __forceinline__ scalar_t apply_clamp(scalar_t x, float min_val) {
    scalar_t min_cast = static_cast<scalar_t>(min_val);
    return (x < min_cast) ? min_cast : x;
}

// Device function to perform division
template <typename scalar_t>
__device__ __forceinline__ scalar_t apply_divide(scalar_t x, float divisor) {
    return x / static_cast<scalar_t>(divisor);
}

// Modular function combining clamp and divide operations
template <typename scalar_t>
__device__ __forceinline__ scalar_t clamp_and_divide(scalar_t x, float min_val, float divisor) {
    return apply_divide(apply_clamp(x, min_val), divisor);
}

// CPU kernel applying the clamp and divide operation on each element
template <typename scalar_t>
void clamp_and_divide_kernel(scalar_t* output,
                             const int64_t numel,
                             float min_val,
                             float divisor) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < numel; i += stride) {
        output[i] = clamp_and_divide(output[i], min_val, divisor);
    }
}

// Forward function performing 3D transposed convolution, then applying the kernel
torch::Tensor forward(torch::Tensor input,
                      int stride,
                      int padding,
                      float min_val,
                      float divisor,
                      torch::Tensor weight,
                      torch::Tensor bias) {
    // Execute 3D transposed convolution via PyTorch
    auto output = torch::conv_transpose3d(input, weight, bias, stride, padding);

    const int threads = 256;
    int blocks = (output.numel() + threads - 1) / threads;
    if (blocks > 65535) {
        blocks = 65535;
    }

    AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "clamp_and_divide", ([&] {
        clamp_and_divide_kernel<scalar_t>(
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