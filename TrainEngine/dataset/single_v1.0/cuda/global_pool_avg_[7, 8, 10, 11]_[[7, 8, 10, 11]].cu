extern "C" __global__ void __launch_bounds__(2) default_function_kernel_1(float* __restrict__ adaptive_pool_avg, float* __restrict__ adaptive_pool_sum) {
  adaptive_pool_avg[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))] = (adaptive_pool_sum[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))] * 9.090909e-03f);
}

extern "C" __global__ void __launch_bounds__(4) default_function_kernel(float* __restrict__ adaptive_pool_sum, float* __restrict__ data) {
  adaptive_pool_sum[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] = 0.000000e+00f;
  for (int rv0 = 0; rv0 < 10; ++rv0) {
    for (int rv1 = 0; rv1 < 11; ++rv1) {
      adaptive_pool_sum[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] = (adaptive_pool_sum[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] + data[((((((int)blockIdx.x) * 440) + (((int)threadIdx.x) * 110)) + (rv0 * 11)) + rv1)]);
    }
  }
}

