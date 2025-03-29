#include <torch/extension.h>
#include <omp.h>
#include <vector>
#include <limits>

// Function to compute argmax over a specified dimension using OpenMP for parallelization
void warp_argmax_cpu(
    const float* x,
    int64_t* indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    #pragma omp parallel for collapse(2)
    for (int outer_idx = 0; outer_idx < outerSize; ++outer_idx) {
        for (int inner_idx = 0; inner_idx < innerSize; ++inner_idx) {
            int start_offset = outer_idx * (dimSize * innerSize) + inner_idx;
            float thread_max = -std::numeric_limits<float>::max();
            int thread_arg = 0;

            for (int d = 0; d < dimSize; ++d) {
                float val = x[start_offset + d * innerSize];
                if (val > thread_max) {
                    thread_max = val;
                    thread_arg = d;
                } else if (val == thread_max && d < thread_arg) {
                    thread_arg = d;
                }
            }

            indices[outer_idx * innerSize + inner_idx] = thread_arg;
        }
    }
}

// Host function to compute argmax
torch::Tensor argmax_forward_cpu(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    const int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dimension for argmax.");

    int outerSize = 1;
    for (int i = 0; i < dim; i++) {
        outerSize *= sizes[i];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int i = dim + 1; i < ndim; i++) {
        innerSize *= sizes[i];
    }

    std::vector<int64_t> out_sizes;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            out_sizes.push_back(sizes[i]);
        }
    }
    
    auto options = torch::TensorOptions().dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    warp_argmax_cpu(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize);

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cpu, "ArgMax CPU forward (OpenMP parallelization)");
}
