#include <torch/extension.h>
#include <omp.h>
#include <cmath>

template <typename scalar_t>
void l2norm_strided_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride,
    const int num_threads) {

    #pragma omp parallel for num_threads(num_threads)
    for (int vector_idx = 0; vector_idx < total_vectors; ++vector_idx) {
        const int base = vector_idx * outer_stride;
        scalar_t thread_sum = 0;

        for (int i = omp_get_thread_num(); i < C; i += num_threads) {
            scalar_t val = input[base + i * stride_C];
            thread_sum += val * val;
        }

        // Reduction using OpenMP reduction clause
        scalar_t block_sum = 0;
        #pragma omp critical
        {
            block_sum = thread_sum;
        }

        // Compute normalization factor
        scalar_t inv_norm = rsqrt(block_sum + 1e-12);

        // Normalize using stride loops
        for (int i = omp_get_thread_num(); i < C; i += num_threads) {
            output[base + i * stride_C] = input[base + i * stride_C] * inv_norm;
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);

    // Number of threads
    const int num_threads = omp_get_max_threads();

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2norm_strided", ([&] {
        l2norm_strided_kernel<scalar_t>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            C,
            total_vectors,
            stride_C,
            outer_stride,
            num_threads
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization with stride optimization");
}