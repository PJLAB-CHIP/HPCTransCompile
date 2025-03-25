#include <torch/extension.h>
#include <omp.h>
#include <stdexcept>

template <typename scalar_t>
void hardtanh_cpu(const scalar_t* __restrict__ x,
                  scalar_t* __restrict__ out,
                  int64_t numel,
                  scalar_t min_val,
                  scalar_t max_val) {
#pragma omp parallel for
  for (int64_t i = 0; i < numel; ++i) {
    scalar_t val = x[i];
    // Clamp between min_val and max_val.
    if (val < min_val) {
      val = min_val;
    } else if (val > max_val) {
      val = max_val;
    }
    out[i] = val;
  }
}

at::Tensor forward_cpu(const at::Tensor& x, float min_val, float max_val) {
  auto out = at::empty_like(x);
  int64_t numel = x.numel();

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hardtanh_cpu", ([&] {
    hardtanh_cpu<scalar_t>(
        x.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        numel,
        static_cast<scalar_t>(min_val),
        static_cast<scalar_t>(max_val)
    );
  }));

  return out;
}

at::Tensor forward(const at::Tensor& x, float min_val, float max_val) {
  if (x.is_cuda()) {
    throw std::invalid_argument("Input tensor must be a CPU tensor");
  }
  return forward_cpu(x, min_val, max_val);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardTanh activation (CPU)");
}