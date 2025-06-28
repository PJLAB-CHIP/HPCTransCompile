extern "C" __global__ void __launch_bounds__(64) default_function_kernel_1(float* __restrict__ adaptive_pool_avg, float* __restrict__ adaptive_pool_sum) {
  adaptive_pool_avg[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = (adaptive_pool_sum[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] / ((float)((((((((int)threadIdx.x) & 7) + 1) % 8) == 0) ? ((((((int)threadIdx.x) & 7) * 17) + 17) >> 3) : (((((((int)threadIdx.x) & 7) * 17) + 17) >> 3) + 1)) - ((((int)threadIdx.x) & 7) * 2))));
}

extern "C" __global__ void __launch_bounds__(64) default_function_kernel(float* __restrict__ adaptive_pool_sum, float* __restrict__ data) {
  adaptive_pool_sum[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = 0.000000e+00f;
  for (int rv1 = 0; rv1 < ((((((((int)threadIdx.x) & 7) + 1) % 8) == 0) ? ((((((int)threadIdx.x) & 7) * 17) + 17) >> 3) : (((((((int)threadIdx.x) & 7) * 17) + 17) >> 3) + 1)) - ((((int)threadIdx.x) & 7) * 2)); ++rv1) {
    adaptive_pool_sum[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = (adaptive_pool_sum[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] + data[((((((int)blockIdx.x) * 136) + ((((int)threadIdx.x) >> 3) * 17)) + ((((int)threadIdx.x) & 7) * 2)) + rv1)]);
  }
}

