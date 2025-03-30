#include <torch/extension.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <omp.h>

template <typename scalar_t>
void argmin_tuned_blocks_kernel(
    const scalar_t* x,
    int64_t* output,
    int K,
    int64_t outer_size,
    int64_t inner_size) {
    
    #pragma omp parallel for
    for (int64_t slice_idx = 0; slice_idx < outer_size * inner_size; ++slice_idx) {
        int64_t outer = slice_idx / inner_size;
        int64_t inner = slice_idx % inner_size;
        
        scalar_t local_min = std::numeric_limits<scalar_t>::max();
        int local_min_idx = 0;
        
        for (int k = 0; k < K; ++k) {
            scalar_t val = x[outer * (K * inner_size) + k * inner_size + inner];
            if (val < local_min) {
                local_min = val;
                local_min_idx = k;
            }
        }
        
        output[slice_idx] = local_min_idx;
    }
}

at::Tensor argmin_cpu_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cpu(), "Input tensor must be a CPU tensor");
    
    int dims = x.dim();
    if (dim < 0) dim += dims;
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }
    int K = static_cast<int>(x.size(dim));
    int64_t inner_size = 1;
    for (int i = dim + 1; i < dims; i++) {
        inner_size *= x.size(i);
    }
    
    std::vector<int64_t> out_sizes;
    for (int i = 0; i < dims; i++) {
        if (i == dim) continue;
        out_sizes.push_back(x.size(i));
    }
    auto output = at::empty(out_sizes, x.options().dtype(at::kLong));
    
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cpu_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_tuned_blocks_kernel<scalar_t>(x_data, output_data, K, outer_size, inner_size);
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmin_cpu_forward, "Argmin forward (CPU)");
}