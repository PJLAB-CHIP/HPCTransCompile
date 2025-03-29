#include <torch/extension.h>
#include <vector>
#include <limits>
#include <omp.h>

// CPU implementation of the min reduction kernel using OpenMP

template <typename scalar_t>
void min_reduce_cpu(const scalar_t* input, scalar_t* output, int outer, int r, int inner) {
  #pragma omp parallel for collapse(2)
  for (int outer_idx = 0; outer_idx < outer; ++outer_idx) {
    for (int inner_idx = 0; inner_idx < inner; ++inner_idx) {
      int base = outer_idx * (r * inner) + inner_idx;
      scalar_t my_min = std::numeric_limits<scalar_t>::max();
      for (int j = 0; j < r; ++j) {
        int pos = base + j * inner;
        scalar_t val = input[pos];
        if (val < my_min) {
          my_min = val;
        }
      }
      output[outer_idx * inner + inner_idx] = my_min;
    }
  }
}

// Forward function: prepares tensor dimensions and calls the CPU kernel

torch::Tensor forward(torch::Tensor input, int64_t dim) {
  if (!input.is_contiguous()) {
    input = input.contiguous();
  }

  int ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // Calculate sizes: outer dimensions, size of reduction dimension (r), and inner dimensions
  int outer = 1;
  for (int i = 0; i < dim; i++) {
    outer *= input.size(i);
  }
  int r = input.size(dim);
  int inner = 1;
  for (int i = dim + 1; i < ndim; i++) {
    inner *= input.size(i);
  }

  // Create the output shape by removing the reduced dimension
  std::vector<int64_t> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (i != dim) {
      output_shape.push_back(input.size(i));
    }
  }
  auto output = torch::empty(output_shape, input.options());

  // Call the CPU kernel
  AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_cpu", ([&] {
    min_reduce_cpu<scalar_t>(input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), outer, r, inner);
  }));

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Min reduction over a specified dimension using OpenMP (CPU)");
}
