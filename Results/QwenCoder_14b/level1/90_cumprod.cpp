#include <torch/extension.h>
#include <omp.h>
#include <vector>

template <typename scalar_t>
void cumprod_min_sync_cpu(
    scalar_t* output,
    const scalar_t* input,
    const int64_t dim_size,
    const int64_t stride,
    const int64_t total_chains) {

    #pragma omp parallel for
    for (int chain_id = 0; chain_id < total_chains; ++chain_id) {
        int batch_idx = chain_id / stride;
        int in_idx = chain_id % stride;
        int64_t base = batch_idx * (dim_size * stride) + in_idx;

        int t = omp_get_thread_num();
        int T = omp_get_num_threads();
        int chunk = (dim_size + T - 1) / T;
        int start = t * chunk;
        int end = start + chunk;
        if (end > dim_size) end = dim_size;

        // First pass: each thread computes the product over its assigned segment
        scalar_t local_prod = 1;
        for (int i = start; i < end; i++) {
            int64_t idx = base + i * stride;
            local_prod *= input[idx];
        }

        // Compute exclusive prefix product offset for this thread
        scalar_t offset = 1;
        for (int i = 0; i < t; i++) {
            offset *= local_prod;
        }

        // Second pass: compute the cumulative product for the assigned segment using the offset
        scalar_t prod = offset;
        for (int i = start; i < end; i++) {
            int64_t idx = base + i * stride;
            prod *= input[idx];
            output[idx] = prod;
        }
    }
}

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

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cpu_min_sync", ([&] {
        cumprod_min_sync_cpu<scalar_t>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            dim_size,
            stride_val,
            total_chains
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cpu_min_sync_forward, "Minimal synchronization cumulative product forward (CPU)");
}
