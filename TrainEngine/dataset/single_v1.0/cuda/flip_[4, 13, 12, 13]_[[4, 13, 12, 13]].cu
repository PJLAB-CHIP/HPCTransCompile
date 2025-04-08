extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ T_reverse_sequence, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 4)) < 507) {
    T_reverse_sequence[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = data[(((((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) % 2028) + 6084) - ((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2)) / 507) * 2028))];
  }
}

