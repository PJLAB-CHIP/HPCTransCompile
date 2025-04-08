extern "C" __global__ void __launch_bounds__(6) default_function_kernel_1(float* __restrict__ adaptive_pool_avg, float* __restrict__ adaptive_pool_sum) {
  if (((((int)blockIdx.x) * 3) + (((int)threadIdx.x) >> 1)) < 19) {
    adaptive_pool_avg[((((int)blockIdx.x) * 6) + ((int)threadIdx.x))] = (adaptive_pool_sum[((((int)blockIdx.x) * 6) + ((int)threadIdx.x))] * 1.250000e-01f);
  }
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ adaptive_pool_sum, float* __restrict__ data) {
  if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 1)) < 19) {
    adaptive_pool_sum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = 0.000000e+00f;
  }
  for (int rv0 = 0; rv0 < 4; ++rv0) {
    for (int rv1 = 0; rv1 < 2; ++rv1) {
      if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) >> 1)) < 19) {
        adaptive_pool_sum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (adaptive_pool_sum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + data[((((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 8)) + (rv0 * 2)) + rv1)]);
      }
    }
  }
}

