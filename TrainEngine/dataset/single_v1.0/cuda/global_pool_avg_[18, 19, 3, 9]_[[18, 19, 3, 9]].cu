extern "C" __global__ void __launch_bounds__(4) default_function_kernel_1(float* __restrict__ adaptive_pool_avg, float* __restrict__ adaptive_pool_sum) {
  if (((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 1)) < 171) {
    adaptive_pool_avg[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] = (adaptive_pool_sum[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] * 3.703704e-02f);
  }
}

extern "C" __global__ void __launch_bounds__(2) default_function_kernel(float* __restrict__ adaptive_pool_sum, float* __restrict__ data) {
  adaptive_pool_sum[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))] = 0.000000e+00f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 9; ++rv1) {
      adaptive_pool_sum[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))] = (adaptive_pool_sum[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))] + data[((((((int)blockIdx.x) * 54) + (((int)threadIdx.x) * 27)) + (rv0 * 9)) + rv1)]);
    }
  }
}

