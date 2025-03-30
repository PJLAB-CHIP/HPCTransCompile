#include <torch/extension.h>
#include <vector>
#include <omp.h>

// Constants defining the tile size (number of output elements processed per block) and the number of threads for reduction per output element
#define TILE 8
#define REDUCE_THREADS 32

// Host function to setup and launch the kernel
torch::Tensor mean_reduce_cpu(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();

    // Get input sizes and compute L (length along reduction dimension), outer_size, and inner_size
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

    // Total number of output elements (after reducing the dimension)
    int64_t N = outer_size * inner_size;
    int stride = inner_size;  // stride to jump across the reduction dimension in input

    // Create a 1D output tensor; later we will reshape it
    auto output = torch::empty({N}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cpu", ([&] {
        const scalar_t* input_data = input.data_ptr<scalar_t>();
        scalar_t* output_data = output.data_ptr<scalar_t>();

        #pragma omp parallel for
        for (int64_t global_output_idx = 0; global_output_idx < N; ++global_output_idx) {
            // Decode the global output index into (outer, inner) indices
            // Input shape: [outer_size, L, inner_size]
            // Here, stride = inner_size
            int outer_idx = global_output_idx / stride;
            int inner_idx = global_output_idx % stride;
            int base_offset = outer_idx * (L * stride) + inner_idx;

            // Each thread accumulates a partial sum over the reduction dimension using a grid-stride loop
            scalar_t sum = static_cast<scalar_t>(0);
            for (int i = 0; i < L; i += REDUCE_THREADS) {
                sum += input_data[base_offset + i * stride];
            }

            // Store the partial sum into shared memory
            output_data[global_output_idx] = sum / static_cast<scalar_t>(L);
        }
    }));

    // Reshape the output to remove the reduced dimension
    sizes.erase(sizes.begin() + dim);
    output = output.view(sizes);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cpu, "Even Workload Mean Reduction (CPU)");
}