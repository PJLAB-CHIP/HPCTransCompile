#include <torch/extension.h>
#include <omp.h>
#include <algorithm>

// CPU kernel function to perform parallel reduction
template <typename scalar_t>
void max_reduce_parallel_cpu(
    const scalar_t* input,
    scalar_t* output,
    const int64_t dim_size,
    const int64_t inner_size,
    const int64_t num_outputs
) {
    #pragma omp parallel for
    for (int out_idx = 0; out_idx < num_outputs; ++out_idx) {
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;
        int64_t base = outer_idx * dim_size * inner_size + inner_idx;

        scalar_t thread_max = input[base];
        bool valid = true;

        for (int j = 0; j < dim_size; ++j) {
            scalar_t val = input[base + j * inner_size];
            if (valid) {
                thread_max = std::max(thread_max, val);
            } else {
                thread_max = val;
                valid = true;
            }
        }

        output[out_idx] = thread_max;
    }
}

// CPU forward function
torch::Tensor max_reduce_cpu_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0)
        dim += input.dim();
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }
    
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    const int64_t dim_size = input.size(dim);
    
    int64_t num_outputs = outer_size * inner_size;
    
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_parallel_forward", ([&] {
        max_reduce_parallel_cpu<scalar_t>(
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