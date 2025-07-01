extern "C" __global__ void __launch_bounds__(48) default_function_kernel_1(float* __restrict__ adaptive_pool_avg, float* __restrict__ adaptive_pool_sum) {
  adaptive_pool_avg[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))] = (adaptive_pool_sum[((((int)blockIdx.x) * 48) + ((int)threadIdx.x))] / (((float)(((((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 3)) & 7) + 1) % 8) == 0) ? (((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 3)) & 7) + 1) >> 3) : ((((((((int)blockIdx.x) * 6) + (((int)threadIdx.x) >> 3)) & 7) + 1) >> 3) + 1))) * ((float)(((((((((int)threadIdx.x) & 7) * 2) + 2) % 8) == 0) ? (((((int)threadIdx.x) & 7) + 1) >> 2) : ((((((int)threadIdx.x) & 7) + 1) >> 2) + 1)) - ((((int)threadIdx.x) & 7) >> 2)))));
}

extern "C" __global__ void __launch_bounds__(32) default_function_kernel(float* __restrict__ adaptive_pool_sum, float* __restrict__ data) {
  adaptive_pool_sum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = 0.000000e+00f;
  for (int rv0 = 0; rv0 < (((((((((int)blockIdx.x) & 1) * 4) + (((int)threadIdx.x) >> 3)) + 1) % 8) == 0) ? ((((((int)threadIdx.x) + 8) >> 5) + (((int)blockIdx.x) & 1)) >> 1) : (((((((int)threadIdx.x) + 8) >> 5) + (((int)blockIdx.x) & 1)) >> 1) + 1)); ++rv0) {
    for (int rv1 = 0; rv1 < (((((((((int)threadIdx.x) & 7) * 2) + 2) % 8) == 0) ? (((((int)threadIdx.x) & 7) + 1) >> 2) : ((((((int)threadIdx.x) & 7) + 1) >> 2) + 1)) - ((((int)threadIdx.x) & 7) >> 2)); ++rv1) {
      adaptive_pool_sum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (adaptive_pool_sum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + data[(((((((int)blockIdx.x) >> 1) * 2) + (rv0 * 2)) + ((((int)threadIdx.x) & 7) >> 2)) + rv1)]);
    }
  }
}

