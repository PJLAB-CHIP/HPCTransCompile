#include <torch/extension.h>
#include <omp.h>
#include <cmath>
#include <vector>

// Define block dimensions for balanced workload distribution
#define OFFSETS_PER_BLOCK 32
#define THREADS_FEATURE 8

template <typename scalar_t>
void rms_norm_even_workload_kernel_cpu(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int total_offsets,  // batch_size * numel_per_batch
    const int num_features,
    const int numel_per_batch,
    const float eps,
    const int batch_size,
    const int numel_per_batch_int
) {
    #pragma omp parallel for
    for (int global_offset = 0; global_offset < total_offsets; ++global_offset) {
        // Determine the batch id and the offset within the batch
        int batch_id = global_offset / numel_per_batch;
        int offset = global_offset % numel_per_batch;
        int base = batch_id * num_features * numel_per_batch;

        // Each thread in the column computes a partial sum over a subset of feature indices
        scalar_t partial_sum = 0;
        for (int f = omp_get_thread_num() % THREADS_FEATURE; f < num_features; f += THREADS_FEATURE) {
            int pos = base + f * numel_per_batch + offset;
            scalar_t val = input[pos];
            partial_sum += val * val;
        }

        // Reduction: sum of squares
        for (int stride = THREADS_FEATURE / 2; stride > 0; stride /= 2) {
            partial_sum += partial_sum;
        }

        // Compute rms
        scalar_t rms = sqrt(partial_sum / num_features + eps);

        // Normalization: each thread in the column normalizes a subset of feature elements
        for (int f = omp_get_thread_num() % THREADS_FEATURE; f < num_features; f += THREADS_FEATURE) {
            int pos = base + f * numel_per_batch + offset;
            output[pos] = input[pos] / rms;
        }
    }
}

// CPU forward function with a 2D block layout for even workload distribution
torch::Tensor rms_norm_cpu_forward_even_workload(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);

    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    // Total number of (batch, offset) pairs to process
    int total_offsets = batch_size * numel_per_batch;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cpu_even_workload", ([&] {
        rms_norm_even_workload_kernel_cpu<scalar_t>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total_offsets,
            num_features,
            numel_per_batch,
            eps,
            batch_size,
            numel_per_batch
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cpu_forward_even_workload, "RMS normalization forward with balanced workload (CPU)");
}