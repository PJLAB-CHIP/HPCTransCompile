#include <torch/extension.h>
#include <vector>
#include <omp.h>

// This function computes the cumulative product along a contiguous dimension with minimal synchronizations.
// Each thread processes one cumulative product chain (row). The work is divided among threads, where each thread
// computes the product of its assigned segment, writes its local product to shared memory, then, after a single
// omp_get_thread_num(), computes an exclusive offset (the product of local products of threads with lower indices).
// Finally, each thread re-reads its assigned segment and writes the cumulative product to global memory.

template <typename scalar_t>
void cumprod_min_sync_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_chains) {

    // Each thread processes one cumulative product chain
    int thread_id = omp_get_thread_num();
    if (thread_id >= total_chains) return;

    int batch_idx = thread_id / stride;
    int in_idx = thread_id % stride;
    int64_t base = batch_idx * (dim_size * stride) + in_idx;

    int chunk = (dim_size + omp_get_max_threads() - 1) / omp_get_max_threads();
    int start = thread_id * chunk;
    int end = start + chunk;
    if (end > dim_size) end = dim_size;

    // Allocate shared memory for local products; only one omp_get_thread_num() is needed here.
    scalar_t local_prod = 1;
    for (int i = start; i < end; i++) {
        int64_t idx = base + i * stride;
        local_prod *= input[idx];
    }

    // Compute exclusive prefix product offset for this thread
    scalar_t offset = 1;
    for (int i = 0; i < thread_id; i++) {
        offset *= local_prod;
    }

    // Compute the cumulative product for the assigned segment using the offset
    scalar_t prod = offset;
    for (int i = start; i < end; i++) {
        int64_t idx = base + i * stride;
        prod *= input[idx];
        output[idx] = prod;
    }
}

// CPU forward function: assumes the cumulative product is along a contiguous dimension.
// The kernel launches one thread per cumulative product chain.

torch::Tensor cumprod_cpu_min_sync_forward(torch::Tensor input, int64_t dim) {
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    auto output = torch::empty_like(input);

    auto sizes = input.sizes();
    auto strides = input.strides();
    int64_t dim_size = sizes[dim];
    int64_t stride_val = strides[dim];
    int64_t total_chains = input.numel() / dim_size;

    // Use OpenMP for parallelization
    #pragma omp parallel
    {
        cumprod_min_sync_kernel<scalar_t>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            dim_size,
            stride_val,
            total_chains
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cpu_min_sync_forward, "Minimal synchronization cumulative product forward (CPU)");
}