#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <omp.h>

// This function performs the exact same operation as the CUDA kernel code.
// It distributes the reduction workload of each output element across multiple threads
// using a block-level parallel reduction. A grid-stride loop over output elements ensures even
// distribution of work and avoids bottlenecks when the number of output elements is large.

template <typename scalar_t>
void max_reduce_parallel_kernel_cpu(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t dim_size,
    const int64_t inner_size,
    const int64_t num_outputs
) {
    #pragma omp parallel for
    for (int64_t out_idx = 0; out_idx < num_outputs; out_idx++) {
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;
        // Compute the starting index for the reduction along the specified dimension
        int64_t base = outer_idx * dim_size * inner_size + inner_idx;

        scalar_t thread_max = input[base];
        for (int64_t j = 1; j < dim_size; j++) {
            scalar_t val = input[base + j * inner_size];
            thread_max = std::max(thread_max, val);
        }

        // Perform tree-based reduction in shared memory
        for (unsigned int s = dim_size / 2; s > 0; s >>= 1) {
            if (thread_max < input[base + s * inner_size]) {
                thread_max = input[base + s * inner_size];
            }
        }

        output[out_idx] = thread_max;
    }
}

// CPU forward function
torch::Tensor max_reduce_cpu_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0)
        dim += input.dim();
    
    // Compute the product of dimensions before 'dim'
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }
    
    // Compute the product of dimensions after 'dim'
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    const int64_t dim_size = input.size(dim);
    
    // Total number of output elements after reducing the 'dim' dimension
    int64_t num_outputs = outer_size * inner_size;
    
    // Prepare output tensor by removing the reduced dimension
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_parallel_forward", ([&] {
        max_reduce_parallel_kernel_cpu<scalar_t>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            inner_size,
            num_outputs
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cpu_forward, "Max reduce forward (CPU) with distributed workload");
}