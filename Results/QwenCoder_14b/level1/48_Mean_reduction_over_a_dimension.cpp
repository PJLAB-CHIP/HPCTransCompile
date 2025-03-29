#include <torch/extension.h>
#include <omp.h>
#include <vector>

#define TILE 8
#define REDUCE_THREADS 32

template <typename scalar_t>
void even_workload_mean_reduce_cpu(
    const scalar_t* input,
    scalar_t* output,
    int L,
    int stride,
    int N
) {
    #pragma omp parallel for collapse(2)
    for (int tile_idx = 0; tile_idx < TILE; ++tile_idx) {
        for (int reduce_idx = 0; reduce_idx < REDUCE_THREADS; ++reduce_idx) {
            int global_output_idx = omp_get_thread_num() * TILE + tile_idx;
            if (global_output_idx >= N) continue;

            int outer_idx = global_output_idx / stride;
            int inner_idx = global_output_idx % stride;
            int base_offset = outer_idx * (L * stride) + inner_idx;

            scalar_t sum = static_cast<scalar_t>(0);
            for (int i = reduce_idx; i < L; i += REDUCE_THREADS) {
                sum += input[base_offset + i * stride];
            }

            int shmem_idx = tile_idx * REDUCE_THREADS + reduce_idx;
            // Simulate shared memory by using a local array
            scalar_t sdata[TILE * REDUCE_THREADS];
            sdata[shmem_idx] = sum;

            #pragma omp barrier

            for (int s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
                if (reduce_idx < s) {
                    sdata[shmem_idx] += sdata[shmem_idx + s];
                }
                #pragma omp barrier
            }

            if (reduce_idx == 0) {
                output[global_output_idx] = sdata[tile_idx * REDUCE_THREADS] / static_cast<scalar_t>(L);
            }
        }
    }
}

torch::Tensor mean_reduce_cpu(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    std::vector<int64_t> sizes = input.sizes().vec();
    int64_t L = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    int64_t N = outer_size * inner_size;
    int stride = inner_size;

    auto output = torch::empty({N}, input.options());

    even_workload_mean_reduce_cpu<scalar_t>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        static_cast<int>(L),
        stride,
        static_cast<int>(N)
    );

    sizes.erase(sizes.begin() + dim);
    output = output.view(sizes);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cpu, "Even Workload Mean Reduction (CPU)");
}
