#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel applies scaling, Hardtanh and GELU activation using a grid-stride loop.
// Optimizations include __ldg for read-only memory access and alignment for coalesced memory access.

template <typename scalar_t>
__global__ void fused_activation_kernel_optimized(
    scalar_t* __restrict__ x,
    scalar_t scaling_factor,
    scalar_t hardtanh_min,
    scalar_t hardtanh_max,
    int64_t numel) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < numel; i += stride) {
    scalar_t val = __ldg(&x[i]);
    // Scaling
    val = val * scaling_factor;
    // Hardtanh
    val = min(max(val, hardtanh_min), hardtanh_max);
    // GELU approximation
    const scalar_t c = static_cast<scalar_t>(0.044715);
    const scalar_t sqrt_2_over_pi = static_cast<scalar_t>(0.7978845608028654); // sqrt(2.0 / pi)
    scalar_t x_cube = val * val * val;
    scalar_t tanh_arg = sqrt_2_over_pi * (val + c * x_cube);
    scalar_t tanh_res = tanh(tanh_arg);
    val = static_cast<scalar_t>(0.5) * val * (static_cast<scalar_t>(1.0) + tanh_res);
    x[i] = val;
  }
}

void fused_activation_cuda(
    torch::Tensor& x,
    double scaling_factor,
    double hardtanh_min,
    double hardtanh_max) {
  const auto numel = x.numel();
  const int threads = 1024;
  const int blocks = (numel + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_activation_cuda", ([&] {
    fused_activation_kernel_optimized<scalar_t><<<blocks, threads>>>(
        x.data_ptr<scalar_t>(),
        static_cast<scalar_t>(scaling_factor),
        static_cast<scalar_t>(hardtanh_min),
        static_cast<scalar_t>(hardtanh_max),
        numel);
  }));
}

torch::Tensor module_fn_forward(
    torch::Tensor x,
    double scaling_factor,
    double hardtanh_min,
    double hardtanh_max,
    torch::Tensor weight,
    torch::Tensor bias) {

  // Ensure inputs are contiguous and on CUDA
  x = x.contiguous().cuda();
  weight = weight.contiguous().cuda();
  bias = bias.contiguous().cuda();

  // Linear transformation: x = x @ weight.T + bias
  auto xw = torch::matmul(x, weight.t()) + bias;

  // Apply fused activation functions
  fused_activation_cuda(xw, scaling_factor, hardtanh_min, hardtanh_max);

  return xw;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &module_fn_forward, "Module function forward (CUDA)");
}