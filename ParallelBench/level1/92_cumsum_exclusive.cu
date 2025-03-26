```cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

template <typename scalar_t>
__global__ void exclusive_cumsum_kernel(
    const scalar_t* input,
    scalar_t* output,
    const int64_t dim_size,
    const int64_t inner_size,
    const int64_t outer_size) {
    
    const int64_t outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t inner_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (outer_idx >= outer_size || inner_idx >= inner_size) return;
    
    const int64_t offset = outer_idx * dim_size * inner_size + inner_idx;
    scalar_t sum = 0;
    
    for (int64_t i = 0; i < dim_size; ++i) {
        const int64_t pos = offset + i * inner_size;
        output[pos] = sum;
        sum += input[pos];
    }
}

torch::Tensor exclusive_cumsum(torch::Tensor input, int64_t dim) {
    // Ensure input is contiguous
    input = input.contiguous();
    
    // Get tensor dimensions
    auto sizes = input.sizes();
    int64_t dim_size = sizes[dim];
    int64_t inner_size = 1;
    int64_t outer_size = 1;
    
    // Calculate inner and outer dimensions
    for (int64_t i = dim + 1; i < sizes.size(); ++i) {
        inner_size *= sizes[i];
    }
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= sizes[i];
    }
    
    // Create output tensor
    auto output = torch::zeros_like(input);
    
    // Launch kernel
    const int threads = 32;
    dim3 blocks((outer_size + threads - 1) / threads, (inner_size + threads - 1) / threads);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "exclusive_cumsum", ([&] {
        exclusive_cumsum_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            inner_size,
            outer_size);
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &exclusive_cumsum, "Exclusive cumulative sum along a dimension");
}
```