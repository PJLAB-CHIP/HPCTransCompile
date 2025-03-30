#include <torch/extension.h>
#include <vector>
#include <omp.h>

// CPU implementation of the warp-level reduction sum kernel
template <typename scalar_t>
void unroll_warp_reduce_sum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t total_outputs) {

    // Calculate global thread id
    int64_t global_thread_id = omp_get_thread_num();
    int warpSize = 32;

    // Total threads available
    int total_threads = omp_get_max_threads();

    // Each thread processes a portion of the output elements
    for (int out_idx = global_thread_id; out_idx < total_outputs; out_idx += total_threads) {
        // Map the output index to the corresponding outer and inner indices
        int outer_idx = out_idx / inner_size;
        int inner_idx = out_idx % inner_size;

        // Compute the base address for the reduction
        int64_t base = outer_idx * reduce_size * inner_size + inner_idx;
        scalar_t sum_val = 0;

        // Each thread accumulates a partial sum over the reduction dimension
        for (int i = 0; i < reduce_size; i++) {
            sum_val += input[base + i * inner_size];
        }

        // Perform warp-level reduction using shuffle down
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_val += sum_val;
        }

        // Thread 0 writes the final result for this output element
        output[out_idx] = sum_val;
    }
}

// CPU wrapper function
torch::Tensor sum_reduce_cpu(torch::Tensor input, int64_t dim) {
    // Adjust for negative dimensions
    if (dim < 0) dim += input.dim();

    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];

    // Compute outer and inner dimensions
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Output tensor: replacing reduction dimension with 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Total number of output elements is outer_size x inner_size
    int64_t total_outputs = outer_size * inner_size;

    // Configure kernel launch parameters using warp-level reduction
    // Each output element is computed by one thread
    int total_threads = omp_get_max_threads();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cpu", ([&] {
        unroll_warp_reduce_sum_kernel<scalar_t>(
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