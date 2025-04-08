extern "C" __global__ void __launch_bounds__(32) default_function_kernel_1(float* __restrict__ adaptive_pool_avg, float* __restrict__ adaptive_pool_sum) {
  adaptive_pool_avg[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = (adaptive_pool_sum[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] / (((float)(((((((((((int)blockIdx.x) & 1) * 4) + (((int)threadIdx.x) >> 3)) + 1) % 8) == 0) ? ((((((int)blockIdx.x) & 1) * 17) + ((((((int)threadIdx.x) >> 3) * 17) + 17) >> 2)) >> 1) : (((((((int)blockIdx.x) & 1) * 17) + ((((((int)threadIdx.x) >> 3) * 17) + 17) >> 2)) >> 1) + 1)) - ((((int)threadIdx.x) >> 3) * 2)) - ((((int)blockIdx.x) & 1) * 8))) * ((float)(((((((((int)threadIdx.x) & 7) * 4) + 4) % 8) == 0) ? ((((((int)threadIdx.x) & 7) * 3) + 3) >> 1) : (((((((int)threadIdx.x) & 7) * 3) + 3) >> 1) + 1)) - (((((int)threadIdx.x) & 7) * 12) >> 3)))));
}

extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ adaptive_pool_sum, float* __restrict__ data) {
  adaptive_pool_sum[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = 0.000000e+00f;
  for (int rv0 = 0; rv0 < ((((((((int)threadIdx.x) >> 3) + 1) % 8) == 0) ? ((((((int)threadIdx.x) >> 3) * 17) + 17) >> 3) : (((((((int)threadIdx.x) >> 3) * 17) + 17) >> 3) + 1)) - ((((int)threadIdx.x) >> 3) * 2)); ++rv0) {
    for (int rv1 = 0; rv1 < (((((((((int)threadIdx.x) & 7) * 4) + 4) % 8) == 0) ? ((((((int)threadIdx.x) & 7) * 3) + 3) >> 1) : (((((((int)threadIdx.x) & 7) * 3) + 3) >> 1) + 1)) - (((((int)threadIdx.x) & 7) * 12) >> 3)); ++rv1) {
      adaptive_pool_sum[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = (adaptive_pool_sum[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] + data[(((((((int)blockIdx.x) * 204) + ((((int)threadIdx.x) >> 3) * 24)) + (rv0 * 12)) + (((((int)threadIdx.x) & 7) * 12) >> 3)) + rv1)]);
    }
  }
}

