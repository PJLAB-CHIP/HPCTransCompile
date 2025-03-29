#include <torch/extension.h>
#include <omp.h>

// CPU implementation of the warp-level reduction kernel
template <typename scalar_t>
void unroll_cpu_reduce_sum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t total_outputs) {

    #pragma omp parallel for collapse(2)
    for (int out_idx = 0; out_idx < total_outputs; ++out_idx) {
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;
        int64_t base = outer_idx * reduce_size * inner_size + inner_idx;
        scalar_t sum_val = 0;

        for (int i = 0; i < reduce_size; ++i) {
            sum_val += input[base + i * inner_size];
        }

        output[out_idx] = sum_val;
    }
}

// CPU wrapper function
torch::Tensor sum_reduce_cpu(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    int64_t total_outputs = outer_size * inner_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cpu", ([&] {
        unroll_cpu_reduce_sum_kernel<scalar_t>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size,
            total_outputs
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cpu, "Sum reduction forward (CPU)");
}
