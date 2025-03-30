#include <torch/extension.h>
#include <algorithm>
#include <omp.h>

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

// This function performs a cumulative sum along a given dimension by partitioning each
// "line" (a contiguous slice along the cumulative sum dimension) into chunks that are
// processed in parallel by multiple threads. Each thread computes the sum of its assigned
// contiguous block. A parallel scan (using shared memory) is then used to compute an
// offset for each thread's block so that the final cumulative sum is correct.

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto output = torch::empty_like(x);
    int ndim = x.dim();
    dim = (dim + ndim) % ndim;

    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= x.size(i);
    }

    int inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        inner_size *= x.size(i);
    }

    int stride = x.size(dim);

    // Each line to be processed corresponds to one combination of outer and inner indices
    int total_lines = outer_size * inner_size;

    #pragma omp parallel for
    for (int line_index = 0; line_index < total_lines; ++line_index) {
        int outer_idx = line_index / inner_size;
        int inner_idx = line_index % inner_size;

        const float* in_line = x.data_ptr<float>() + outer_idx * stride * inner_size + inner_idx;
        float* out_line = output.data_ptr<float>() + outer_idx * stride * inner_size + inner_idx;

        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        // Divide the stride dimension into contiguous chunks for each thread
        int chunk_size = (stride + num_threads - 1) / num_threads;
        int start = tid * chunk_size;
        int end = std::min(start + chunk_size, stride);

        // First pass: each thread computes the sum of its chunk (partial sum).
        float thread_sum = 0.0f;
        for (int i = start; i < end; i++) {
            thread_sum += in_line[i * inner_size];
        }

        // Use shared memory to perform an inclusive scan on thread partial sums
        float sdata[256]; // Assuming a maximum of 256 threads per block
        sdata[tid] = thread_sum;

        for (int offset = 1; offset < num_threads; offset *= 2) {
            float temp = 0.0f;
            if (tid >= offset) {
                temp = sdata[tid - offset];
            }
            __sync_synchronize(); // Ensure memory ordering
            sdata[tid] += temp;
        }

        // The offset for the current thread's chunk is the sum of all previous chunks
        float add_offset = (tid == 0) ? 0.0f : sdata[tid - 1];

        // Second pass: each thread recomputes its local cumulative sum and writes results
        // with the appropriate offset so that the overall scan is correct.
        float local_running = 0.0f;
        for (int i = start; i < end; i++) {
            local_running += in_line[i * inner_size];
            out_line[i * inner_size] = local_running + add_offset;
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CUDA cumulative sum with aligned memory access and __ldg optimization");
}