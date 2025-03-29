#include <torch/extension.h>
#include <cmath>
#include <omp.h>

template <typename scalar_t>
void rms_norm_cpu_forward_even_workload(
    const scalar_t* input,
    scalar_t* output,
    const int total_offsets,
    const int num_features,
    const int numel_per_batch,
    const float eps
) {
    #pragma omp parallel for collapse(2)
    for (int global_offset = 0; global_offset < total_offsets; ++global_offset) {
        int batch_id = global_offset / numel_per_batch;
        int offset = global_offset % numel_per_batch;
        int base = batch_id * num_features * numel_per_batch;

        scalar_t partial_sum = 0;
        for (int f = 0; f < num_features; ++f) {
            int pos = base + f * numel_per_batch + offset;
            scalar_t val = input[pos];
            partial_sum += val * val;
        }

        scalar_t sumsq = partial_sum;
        scalar_t rms = std::sqrt(sumsq / num_features + eps);

        for (int f = 0; f < num_features; ++f) {
            int pos = base + f * numel_per_batch + offset;
            scalar_t val = input[pos];
            output[pos] = val / rms;
        }
    }
}

torch::Tensor rms_norm_cpu_forward(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);

    const int batch_size = input.size(0);
    const int num_features = input.size(1);

    int numel_per_batch = 1;
    for (int i = 2; i < input.dim(); i++) {
        numel_per_batch *= input.size(i);
    }

    int total_offsets = batch_size * numel_per_batch;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cpu_even_workload", ([&] {
        rms_norm_cpu_forward_even_workload<scalar_t>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total_offsets,
            num_features,
            numel_per_batch,
            eps
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cpu_forward, "RMS normalization forward with balanced workload (CPU)");
}
